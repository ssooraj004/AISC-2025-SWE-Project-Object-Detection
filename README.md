# AISC 2025 SWE Project — Real-Time Object Detection for the Visually Impaired

This project focuses on building an AI-powered real-time object detection system using YOLOv5 to assist visually impaired individuals in navigating their surroundings. The system is designed to provide immediate feedback through audio cues based on detected objects in the user's environment.

## Project Overview

Our pipeline consists of four main stages:

1. **Data Collection** — Curated and collected custom image datasets relevant to common indoor and outdoor environments.
2. **Image Normalization** — Applied preprocessing techniques (resizing, normalization, augmentation) for model robustness.
3. **Model Training** — Experimented with Faster R-CNN and YOLOv5; YOLOv5 was selected for its speed and real-time performance.
4. **Deployment** — Integrated the trained model with a live camera feed, providing audio feedback on detected objects.

## Models Considered

### YOLOv5 (Selected)
- One-stage detector: fast and accurate.
- High FPS for real-time applications.
- Easy to deploy on CPU or edge devices.
- Compatible with Torch and ONNX export.

### Faster R-CNN
- Two-stage detector: high precision.
- Better for small object detection.
- Slower inference, not suitable for real-time performance.

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/ssooraj004/AISC-2025-SWE-Project-Object-Detection.git
cd AISC-2025-SWE-Project-Object-Detection
pip install -r requirements.txt