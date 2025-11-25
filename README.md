# Shoplifting-Detection-PS3

## Introduction
This project implements a **pose-based shoplifting detection system** for surveillance video.  
Rather than analysing raw RGB frames, the pipeline extracts **17 human body keypoints per frame** using YOLO-Pose and feeds pose sequences into a lightweight **Spatio-Temporal Graph (STG) model**. The STG model learns spatial skeleton relationships and temporal motion to classify actions as:

- **Normal**
- **Shoplifting** (bag or pocket)

**Sliding-window inference:**
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

![Model Architecture](assets/model.png)

---

## Pre-requisites

- **Python:** 3.12  
- Install required packages:
```bash
pip install ultralytics torch torchvision torchaudio
pip install numpy matplotlib opencv-python scikit-learn tqdm seaborn
```

## Running Instructions

### **Using the Pre-Trained Model (Recommended)**

 1. Open **`Video-Prediction.py`**.
    
 2. Set the model path:

   ```python
   MODEL_NAME = "./models/Pre-Trained/sgt_model_2class_bs32.pth"
   ```

 3. Set the video path
   ```python
   VIDEO_PATH="path_to_your_video"
   ```

 4. Run Video-Prediction.py

### **Training the Model (If You Want to Verify Metrics)**
 1. Download the JSON Dataset
    https://drive.google.com/drive/folders/1kj3KtDQp_N1juB0cxY-QjyZqfp3mQ7W0 (dataset link)
    
 2. Place the folder Dataset-JSON inside the project root directory.Final Structure should look like this
    Shoplifting-Detection-PS3/
     ├── Dataset-JSON/
     ├── Pre-Processing/
     ├── Training-Script/
     ├── models/
     └── Video-Prediction.py
    
 3. Navigate to the preprocessing directory and run JSON-numpy.py
    ```bash
    cd Pre-Processing
    python3 JSON-numpy.py
    ```
 4. Navigate to Training-Script and run training.py
    ```bash
    cd Training-Script
    python3 training.py
    ```
 5. Confusion Matrix and Loss(validation and training) v/s Epochs
    ![Confusion Matrix](assets/confusion_matrix.png)
    ![Loss Curve](assets/trainingloss_validationloss.png)
    
    

