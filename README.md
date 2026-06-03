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
* Mirror card logic support
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

Some files contain paths configured for my local machine.

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

## Step 2 - Configure Screen Region

Run:

```bash
python src/get_region.py
```

Select your Clash Royale game window and note the coordinates.

Update the coordinates inside:

```text
src/motion_detect_demo.py
```

Example:

```python
monitor = {
    "top": 56,
    "left": 1051,
    "width": 919,
    "height": 1318
}
```

These values are different for every computer.

---

## Step 3 - Run ArenaSense

After training and configuring screen coordinates:

```bash
python src/motion_detect_demo.py
```

The system will:

1. Capture the game window
2. Detect troop deployments
3. Classify detected cards
4. Apply elixir logic
5. Estimate enemy elixir in real time

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
│   ├── get_region.py
│   ├── motion_detect_demo.py
│   └── train_card_classifier.py
│
├── requirements.txt
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
