# Shoplifting-Detection-PS3

## Introduction
This project implements a **pose-based shoplifting detection system** for surveillance video.  
Rather than analysing raw RGB frames, the pipeline extracts **17 human body keypoints per frame** using YOLO-Pose and feeds pose sequences into a lightweight **Spatio-Temporal Graph (STG) model**. The STG model learns spatial skeleton relationships and temporal motion to classify actions as:

- **Normal**
- **Shoplifting** (bag or pocket)

**Sliding-window inference (important):**
- **Window size:** 240 frames  
- **Stride:** 100 frames  
The sliding window ensures temporal continuity and overlapping context between windows. Each window is classified independently; final video-level decision can be obtained by majority voting across windows.

---

## Model Architecture
High-level components:
- Graph convolution over skeleton nodes (17 joints)
- Temporal convolutional layers to capture motion (two STGCN blocks)
- Adaptive global average pooling
- Fully connected classification head

![Model Architecture](assets/model_architecture.png)

---

## Pre-requisites

- **Python:** 3.12  
- Install required packages:
```bash
pip install ultralytics torch torchvision torchaudio
pip install numpy matplotlib opencv-python scikit-learn tqdm seaborn
