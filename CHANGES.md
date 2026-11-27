# What was wrong and what changed

## Why cards were detected but elixir was not deducted
1. **Name mismatch.** The cost table keys were normalised (`baby_dragon`) but the model's class
   names were looked up raw (`baby dragon`), so 70 of 105 classes hit `.get(card, 0)` and cost 0.
2. **Regen refilled every deduction.** Regen only ran while elixir < 10 and the timer froze at max.
   After a play, the stale timer made the next frame "gain" ~10 elixir instantly (6 -> 10 in 0.1 s).
3. **Incomplete table.** Hog Rider, Mega Knight, Wizard, Sparky, Mother Witch, Goblin Demolisher... had
   no entry at all. Monk cost was 4 (is 5).
Fixes: `src/elixir_tracker.py` (symmetric normalisation, full table, float elixir, always-on regen,
piecewise x1/x2/x3, unknown cards assume 4 and are flagged).

## Other flaws fixed
- Motion ran on the whole screen (own troops, UI, towers). Now restricted to the enemy half (`--roi`).
- Crop used unpadded w/h after shifting x/y (off-centre crop). Now padded on all sides and clamped.
- Debounce needed 4 identical predictions in a row (rare on moving sprites). Now 3 votes in 1.2 s.
- Same troop re-detected -> double deduction / false "MIRRORED". Now 4 s same-card lockout.
- Match clock started when the script started. Now `s` re-syncs; `--start-at` for clips.
- Hard-coded `C:\ElixerTracker` paths; inference downloaded ImageNet weights; `requirements.txt`
  location/pyautogui missing; wrong filename printed by get_region.py.
- Training: frozen backbone's BatchNorm was still updating in train mode; test split never evaluated.
- Added `--video`, `--headless`, tests, `check_costs.py`.

## Known limitations (not fixable by code alone)
- The classifier was trained on card **icons**; it is applied to in-arena troop sprites. If live accuracy
  is poor, retrain on arena crops from your own recordings.
- Costs change with balance patches; `check_costs.py` only checks coverage, not correctness.
- Mirror inference is heuristic (`--no-mirror` to disable). Spells cast outside the ROI are missed.

# Round 2: wrong card predictions (e.g. only "spear goblins" / "goblin giant")
Cause: the model was trained on card **icons** but sees **arena sprites + random motion**, so it forces
every motion blob into the nearest icon class. Added tooling to train on your own screen:
- `--deck "a,b,c,..."`   restrict predictions to the opponent's 8 cards (probabilities renormalised).
- `--save-crops DIR`      save every motion crop while running.
- `src/label_crops.py`    keyboard labelling (1-9 = deck card, 0 = background / not a deployment).
- `src/finetune_arena.py` fine-tunes your checkpoint on those crops and adds a `background` class, which the
                          tracker then ignores (no deduction).
- `--model PATH`          run with the fine-tuned checkpoint.
- Use `--no-mirror` until predictions are reliable (wrong predictions create false "MIRRORED").

# Round 3
- Affordability rule: a confirmed card that costs more than the enemy's estimated elixir (+1.0 margin,
  `ELIXIR_MARGIN`) is ignored. Trade-off: if the estimate has drifted too LOW, real plays are blocked
  until it regenerates. Use `+` / `-` / `s` to correct the estimate.
- Removed the `--record` video-writer feature (record with OBS / Win+G instead).
- Added `.gitignore` (videos, crops, data, venv).
