# BabyNest — AI-Powered Infant Monitoring System

An intelligent infant monitoring system that uses deep learning-based image classification to detect infant states (sleeping, crying, awake, empty cradle) from camera feeds, and sends real-time push notifications to caregivers when distress is detected.

## Overview
BabyNest combines a computer vision pipeline with a real-time notification system to provide continuous, automated infant supervision. The core model is built on an EfficientNetB0 backbone (transfer learning from ImageNet), fine-tuned to classify infant states from camera images.

## Key Results
- **Classes (6):** anger, awake, cry, empty_cradle, sleeponback, sleeponstomach
- **Accuracy:** ~90%
- Trained for 20 epochs using transfer learning on GPU-accelerated hardware

## Architecture
- **Backbone:** EfficientNetB0 (frozen, pretrained on ImageNet)
- **Custom head:** GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.3) → Dense(6, Softmax)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-2)
- **Regularization:** In-model data augmentation (RandomFlip, RandomRotation, RandomBrightness), Dropout, class-weighted loss for imbalance
- **Training:** 20 epochs, GPU-accelerated

## Notification System
When the model detects a distress state (e.g., crying) above a confidence threshold, an alert is dispatched via the Pushbullet API to the caregiver's device, including a live webcam capture and predicted class/confidence.

## Tech Stack
Python, TensorFlow/Keras, OpenCV, Tkinter, Pushbullet API, scikit-learn, Matplotlib/Seaborn

## Project Structure
```
babynest/
├── src/
│   ├── babynest_fixed.ipynb    # Model training notebook
│   └── notification_app.py     # GUI + notification integration
├── requirements.txt
└── README.md
```

## Setup & Run
```bash
pip install -r requirements.txt

# Set your Pushbullet token as an environment variable (never hardcode it)
export PUSHBULLET_TOKEN="your_token_here"   # on Windows: set PUSHBULLET_TOKEN=your_token_here

python src/notification_app.py
```

## Future Work
- Multi-modal fusion with audio-based cry classification
- LoRA fine-tuning for on-device personalization
- Migration to higher-capacity EfficientNet variants (B1–B4)
