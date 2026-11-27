# Elixer Counter - Real-Time Enemy Elixir Tracker

Elixer Counter is a computer vision and machine learning project that detects Clash Royale card deployments and estimates enemy elixir in real time using motion detection, image classification, and game-state logic.

## YouTube Walkthrough

Watch the complete project walkthrough here:

https://youtu.be/bKbxBAIL_XY?si=hTeu8PayRhfHZTyp

## Disclaimer

This project is for educational and research purposes only.

All Clash Royale assets, artwork, characters, and intellectual property belong to Supercell.

This project is not affiliated with, endorsed by, or sponsored by Supercell.

## Features

* Real-time game screen capture
* Motion-based troop deployment detection
* MobileNetV2 image classification
* Enemy elixir estimation
* Real-time prediction overlay

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* MobileNetV2
* NumPy
* MSS
* Computer Vision
* Deep Learning

---

## Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/hahafirst/elixercounter-clash-royale-card-dataset

Extract the dataset into:

```text
data/icons/dataset
```

Expected structure:

```text
data/
└── icons/
    └── dataset/
        ├── train/
        ├── val/
        └── test/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/manav731/elixir-counter.git
cd elixir-counter
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Important Path Configuration

Paths are now relative to the repo. Override with the `ELIXIR_DATA` / `ELIXIR_MODEL` environment variables.

(Legacy note, no longer needed:)

Example:

```python
DATA_DIR = r"C:\ElixerTracker\data\icons\dataset"
```

or

```python
MODEL_OUT = r"C:\ElixerTracker\models\card_classifier.pth"
```

If these paths do not exist on your machine, change them according to where you stored the project.

Example:

```python
DATA_DIR = r"D:\Projects\ArenaSense\data\icons\dataset"
```

```python
MODEL_OUT = r"D:\Projects\ArenaSense\models\card_classifier.pth"
```

Make sure all dataset and model paths point to valid locations before running the project.

---

## Step 1 - Train the Model

Run:

```bash
python src/train_card_classifier.py
```

The script will:

* Load the dataset
* Apply data augmentation
* Train MobileNetV2
* Save the trained model automatically

Output:

```text
models/card_classifier.pth
```

Training time depends on your hardware.

---

## Step 2 - Find Your Screen Region

```bash
python src/get_region.py
```

It prints a ready-to-paste `--region TOP,LEFT,WIDTH,HEIGHT` argument. No file editing needed.

---

## Step 3 - Run the Elixir Counter

Live (start it as the battle begins, press `s` at the "Fight" moment to sync the clock):

```bash
python src/motion_detect_demo.py --region 56,1051,919,1318
```

Or on a recorded clip (deterministic, ideal for demos):

```bash
python src/motion_detect_demo.py --video gameplay.mp4 --start-at 3.0
```

Hotkeys: `s` re-sync battle start, `+` / `-` nudge elixir when it drifts, `q` quit.
Tune `--roi 0.12,0.42` so the yellow "enemy zone" box covers the opponent's half of the arena
(down to the river). Only motion inside it is treated as an enemy deployment.

Sanity checks: `python -m pytest tests` and `python src/check_costs.py` (run the latter after retraining).

---

## Using Your Own Dataset

You may replace the provided dataset with your own.

Required structure:

```text
dataset/
├── train/
├── val/
└── test/
```

Each split should contain folders named after card classes:

```text
train/
├── bomber/
├── knight/
├── pekka/
└── ...
```

After replacing the dataset, retrain using:

```bash
python src/train_card_classifier.py
```

---

## Repository Structure

```text
elixir-counter
│
├── models
│   └── card_classifier.pth
│
├── src
│   ├── card_classifier.py
│   ├── elixir_tracker.py   # elixir/cost/mirror logic (unit-tested)
│   ├── check_costs.py
│   ├── get_region.py
│   ├── motion_detect_demo.py
│   └── train_card_classifier.py
│
├── requirements.txt
├── tests/
├── LICENSE.txt
└── README.md
```

---

## Technical Skills Demonstrated

* Computer Vision
* Deep Learning
* Transfer Learning
* Motion Detection
* Real-Time Systems
* Image Classification
* PyTorch
* OpenCV
* ML System Design

---

## Author

Manav Mehta

If you use this project, please consider giving the repository a star.

---

## License

Released under the MIT License.


---

## Fixing wrong card predictions (train on your own screen)

The shipped model was trained on card icons. For reliable results, fine-tune it on crops from your setup:

```bash
# 1. Collect crops while you play / watch a battle (or on a recorded clip with --video)
python src/motion_detect_demo.py --region T,L,W,H --save-crops crops --no-mirror

# 2. Label them. Use the OPPONENT'S deck from that battle (shown in the in-game battle log)
python src/label_crops.py --crops crops --deck "hog rider,musketeer,fireball,ice spirit,skeletons,cannon,ice golem,log"
#    keys: 1-8 = card, 0 = background/not a deployment, space = skip, u = undo, q = quit

# 3. Repeat 1-2 for several battles (aim for 30+ crops per card, 100+ background), then train
python src/finetune_arena.py --data data/arena --out models/card_classifier_arena.pth

# 4. Run with the new model and the opponent's deck
python src/motion_detect_demo.py --region T,L,W,H --model models/card_classifier_arena.pth --deck "..."
```
