import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from elixir_tracker import (CardVoter, ElixirTracker, SEC_PER_ELIXIR, lookup_cost, normalize_name)

CLASSES = open(os.path.join(os.path.dirname(__file__), "classes.txt")).read().split("\n")


def test_every_model_class_has_a_cost():
    assert len(CLASSES) == 105
    assert [c for c in CLASSES if not lookup_cost(c)[1]] == []


@pytest.mark.parametrize("name,cost", [
    ("baby dragon", 4), ("hog rider", 4), ("mega knight", 7), ("p.e.k.k.a", 7),
    ("mini p.e.k.k.a", 4), ("x-bow", 6), ("goblin mchine", 5), ("knight", 3)])
def test_multiword_and_special_names(name, cost):
    assert lookup_cost(name) == (cost, True)


def test_unknown_card_is_not_free():
    cost, known = lookup_cost("totally new card")
    assert cost > 0 and not known


def test_normalize_is_symmetric():
    assert normalize_name("Mini P.E.K.K.A") == normalize_name("mini_p.e.k.k.a") == "minipekka"


def test_regression_deduction_not_refilled_at_max():
    """Original bug: at 10 elixir the regen timer froze, so the next frame refilled to 10."""
    t = ElixirTracker(now=0.0)
    t.advance(30.0)                       # idle until capped
    assert t.elixir == 10.0
    t.play("hog rider", 30.0)
    t.advance(30.1)
    assert t.elixir == pytest.approx(6.0 + 0.1 / SEC_PER_ELIXIR)
    t.advance(35.0)
    assert t.elixir == pytest.approx(6.0 + 5.0 / SEC_PER_ELIXIR)


def test_regen_rates_and_boundary_split():
    t = ElixirTracker(now=0.0)
    t.elixir = 0.0
    t.advance(118.0)                                  # x1 -> capped, so test increments separately
    t.elixir = 0.0; t.last_t = 119.0
    t.advance(121.0)                                  # 1s at x1 + 1s at x2
    assert t.elixir == pytest.approx(1 / 2.8 + 2 / 2.8)
    t.elixir = 0.0; t.last_t = 239.0
    t.advance(241.0)                                  # 1s at x2 + 1s at x3
    assert t.elixir == pytest.approx(2 / 2.8 + 3 / 2.8)


def test_starts_at_five_and_floors_at_zero():
    t = ElixirTracker(now=0.0)
    assert t.elixir == 5.0
    t.play("golem", 0.0)                              # 8 > 5
    assert t.elixir == 0.0


def test_same_deployment_not_double_counted():
    t = ElixirTracker(now=0.0)
    assert t.play("giant", 1.0) is not None
    assert t.play("giant", 2.5) is None               # still the same troop on screen
    assert t.elixir == pytest.approx(2.5 / 2.8)       # 5 start + 2.5s regen - one 5-cost deduction


def test_mirror_adds_one_and_can_be_disabled():
    t = ElixirTracker(now=0.0); t.elixir = 10.0
    t.play("knight", 10.0); ev = t.play("knight", 20.0)
    assert ev.mirrored and ev.cost == 4
    t2 = ElixirTracker(now=0.0, mirror=False); t2.elixir = 10.0
    t2.play("knight", 10.0); ev2 = t2.play("knight", 20.0)
    assert not ev2.mirrored and ev2.cost == 3


def test_different_card_is_not_mirror():
    t = ElixirTracker(now=0.0)
    t.play("knight", 1.0); ev = t.play("archers", 2.0)
    assert not ev.mirrored


def test_voter_majority_beats_noise():
    v = CardVoter(window_sec=1.2, min_votes=3)
    assert v.add("giant", 0.0) is None
    assert v.add("golem", 0.1) is None
    assert v.add("giant", 0.2) is None
    assert v.add("giant", 0.3) == "giant"


def test_voter_window_expires():
    v = CardVoter(window_sec=1.0, min_votes=3)
    v.add("giant", 0.0); v.add("giant", 0.1)
    assert v.add("giant", 5.0) is None
