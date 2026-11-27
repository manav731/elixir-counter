"""
Elixir game-state logic. Pure Python (no torch / cv2 / mss) so it can be unit-tested.

Fixes vs. the original motion_detect_demo.py
  * Cost lookup normalises BOTH sides (class name and table key) the same way, so
    "baby dragon", "Baby Dragon", "baby_dragon" all resolve. (Original normalised only
    the table, so 70 of the model's 105 classes returned cost 0 => no deduction.)
  * Table covers every class the model outputs (hog rider, mega knight, wizard, ...).
  * Elixir is a float, accrued continuously, and accrual is NEVER paused at max, so a
    deduction made while at 10 elixir is no longer instantly "refilled".
  * Regen is integrated piecewise across the x1 -> x2 -> x3 boundaries.
  * Unknown cards fall back to UNKNOWN_CARD_COST instead of silently costing 0.
  * Same-deployment lockout + majority-vote debouncer.
"""
import re
from collections import Counter, deque
from dataclasses import dataclass

START_ELIXIR = 5.0
MAX_ELIXIR = 10.0
SEC_PER_ELIXIR = 2.8      # 1 elixir / 2.8 s at normal speed
DOUBLE_AT = 120.0         # seconds since match start: last minute of regulation (x2)
TRIPLE_AT = 240.0         # 1 min into overtime (x3)
UNKNOWN_CARD_COST = 4     # used (and flagged) if a class has no table entry
SAME_CARD_LOCKOUT = 4.0   # s: same card re-detected inside this window = same deployment


def normalize_name(name: str) -> str:
    """'Mini P.E.K.K.A' / 'mini p.e.k.k.a' / 'mini_pekka' -> 'minipekka'."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


# Keys use the model's class-folder spellings (see models/card_classifier.pth).
# Evolutions / heroes cost the same as the base card.
# NOTE: verify against the current game balance patch; a few newer cards are marked (?).
CARD_COSTS_RAW = {
    "archers": 3, "arrows": 3, "baby dragon": 4, "balloon": 5, "bandit": 3,
    "barbarian hut": 7, "barbarians": 5, "bats": 2, "battle healer": 4,
    "battle ram": 4, "berserker": 2, "bomb tower": 4, "bomber": 2, "bowler": 5,
    "cannon": 3, "cannon cart": 5, "dark prince": 4, "dart goblin": 3,
    "electro dragon": 5, "electro giant": 7, "electro spirit": 1,
    "electro wizard": 4, "elite barbarians": 6, "elixir collector": 6,
    "elixir golem": 3, "empress": 6,               # (?) alias of spirit empress
    "evo barbarian": 5, "evo furnace": 4, "evo hunter": 4,
    "evo skeleton barrel": 3, "executioner": 5, "fire spirit": 1, "fireball": 4,
    "firecracker": 3, "fisherman": 3, "flying machine": 4, "furnace": 4,
    "giant": 5, "giant skeleton": 6, "giant snowball": 2, "goblin barrel": 3,
    "goblin cage": 4, "goblin demolisher": 4, "goblin gang": 3, "goblin giant": 6,
    "goblin hut": 5, "goblin machine": 5, "goblins": 2, "goblinstein": 5,   # (?)
    "golden knight": 4, "golem": 8, "guards": 3, "heal spirit": 1,
    "hero musketeer": 4, "hog rider": 4, "hunter": 4, "ice golem": 2,
    "ice spirit": 1, "ice wizard": 3, "inferno dragon": 4, "inferno tower": 5,
    "knight": 3, "lava hound": 7, "little prince": 3, "lumberjack": 4,
    "magic archer": 4, "mega knight": 7, "mega minion": 3, "mini p.e.k.k.a": 4,
    "minion horde": 5, "minions": 3, "monk": 5, "mortar": 4, "mother witch": 4,
    "musketeer": 4, "night witch": 4, "p.e.k.k.a": 7, "phoenix": 4, "prince": 5,
    "princess": 3, "ram rider": 5, "rascals": 5, "rocket": 6, "royal ghost": 3,
    "royal giant": 6, "royal hogs": 5, "royal recruits": 7, "rune giant": 4,
    "skeleton army": 3, "skeleton barrel": 3, "skeleton dragons": 4,
    "skeleton king": 4, "skeletons": 1, "sparky": 6, "spear goblins": 2,
    "spirit empress": 6, "suspicious bush": 2, "tesla": 4, "three musketeers": 9,
    "tombstone": 3, "valkyrie": 4, "wall breakers": 2, "witch": 5, "wizard": 5,
    "x-bow": 6, "zappies": 4,
}

# The dataset folder has a typo ("goblin mchine"); map it onto the correct key.
ALIASES = {"goblinmchine": "goblinmachine"}

CARD_COSTS = {normalize_name(k): v for k, v in CARD_COSTS_RAW.items()}


def lookup_cost(card: str):
    """Return (cost, known). Never returns 0 for an unknown card."""
    key = normalize_name(card)
    key = ALIASES.get(key, key)
    if key in CARD_COSTS:
        return CARD_COSTS[key], True
    return UNKNOWN_CARD_COST, False


@dataclass
class PlayEvent:
    card: str
    cost: int
    known: bool
    mirrored: bool
    elixir_before: float
    elixir_after: float
    t: float


class ElixirTracker:
    """Estimates the opponent's elixir. All times are seconds on any monotonic clock
    (wall clock for live capture, frame_index/fps for recorded video)."""

    def __init__(self, now: float = 0.0, mirror: bool = True):
        self.mirror = mirror
        self.history = []
        self.reset(now)

    def reset(self, now: float):
        self.match_start = now
        self.last_t = now
        self.elixir = START_ELIXIR
        self.last_card_key = None
        self.last_card_t = -1e9
        self.history = []

    # ---- regen ---------------------------------------------------------
    def multiplier_at(self, t: float) -> int:
        e = t - self.match_start
        return 3 if e >= TRIPLE_AT else 2 if e >= DOUBLE_AT else 1

    def phase(self, now: float) -> str:
        return f"x{self.multiplier_at(now)}"

    def advance(self, now: float):
        """Accrue elixir from last_t to now, splitting at the x2 / x3 boundaries.
        Always runs (even at max) so last_t never goes stale."""
        t = self.last_t
        if now <= t:
            return
        while t < now:
            e = t - self.match_start
            if e < DOUBLE_AT:
                boundary, mult = self.match_start + DOUBLE_AT, 1
            elif e < TRIPLE_AT:
                boundary, mult = self.match_start + TRIPLE_AT, 2
            else:
                boundary, mult = float("inf"), 3
            seg_end = min(now, boundary)
            self.elixir = min(MAX_ELIXIR, self.elixir + (seg_end - t) * mult / SEC_PER_ELIXIR)
            t = seg_end
        self.last_t = now

    # ---- card plays ----------------------------------------------------
    def play(self, card: str, now: float):
        """Register a confirmed enemy card play. Returns PlayEvent, or None if ignored
        as a duplicate of the deployment we just counted."""
        self.advance(now)
        key = ALIASES.get(normalize_name(card), normalize_name(card))
        if key == self.last_card_key and (now - self.last_card_t) < SAME_CARD_LOCKOUT:
            return None
        cost, known = lookup_cost(card)
        # A card can't be re-played until 4 others were played, so the same card
        # appearing straight after itself (past the lockout) is a Mirror: cost + 1.
        mirrored = self.mirror and key == self.last_card_key
        if mirrored:
            cost += 1
        before = self.elixir
        self.elixir = max(0.0, self.elixir - cost)
        self.last_card_key, self.last_card_t = key, now
        ev = PlayEvent(card, cost, known, mirrored, before, self.elixir, now - self.match_start)
        self.history.append(ev)
        return ev

    def adjust(self, delta: float, now: float):
        """Manual correction from the keyboard (+/-)."""
        self.advance(now)
        self.elixir = max(0.0, min(MAX_ELIXIR, self.elixir + delta))


class CardVoter:
    """Majority vote over a short time window (replaces 'N identical predictions in a row',
    which almost never happens while a troop is moving)."""

    def __init__(self, window_sec: float = 1.2, min_votes: int = 3):
        self.window, self.min_votes = window_sec, min_votes
        self.votes = deque()

    def add(self, card: str, now: float):
        self.votes.append((now, card))
        while self.votes and now - self.votes[0][0] > self.window:
            self.votes.popleft()
        card_, n = Counter(c for _, c in self.votes).most_common(1)[0]
        return card_ if n >= self.min_votes else None

    def clear(self):
        self.votes.clear()
