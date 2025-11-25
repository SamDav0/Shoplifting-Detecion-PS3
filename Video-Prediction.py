import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import os

# ================================
# 1. SGT Model Classes
# ================================
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, A):
        x = torch.einsum('nctv,vw->nctw', x, A)
        return self.conv(x)


class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1),
                      ((kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

    def forward(self, x, A):
        x = self.gcn(x, A)
        x = self.tcn(x)
        return x


class SGT_Model(nn.Module):
    def __init__(self, num_classes, in_channels, num_nodes):
        super().__init__()
        self.stgcn1 = STGCN_Block(in_channels, 64, kernel_size=9, stride=1, dropout=0.3)
        self.stgcn2 = STGCN_Block(64, 128, kernel_size=9, stride=2, dropout=0.3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, A):
        x = self.stgcn1(x, A)
        x = self.stgcn2(x, A)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ================================
# 2. YOLO Keypoint Extraction
# ================================
def extract_keypoints_yolo(video_path):
    model = YOLO("yolo11n-pose.pt")
    video_data = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    print(f"Video FPS detected: {fps}")
    print("Extracting keypoints using YOLO...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        try:
            pts = results[0].keypoints.xy[0]
            if pts.shape[0] >= 17:
                video_data.append(pts[:17].cpu().numpy())
            else:
                video_data.append(np.zeros((17, 2)))
        except:
            video_data.append(np.zeros((17, 2)))

    cap.release()
    return video_data, fps


# ================================
# 3. Prediction Settings
# ================================
VIDEO_PATH = ''     # ← update your video
MODEL_DIR = './models'               # folder for saved models
MODEL_NAME = ''
SGT_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

MAX_FRAMES = 240
STRIDE = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Normal', 'Shoplifting']


# ================================
# 4. Run Prediction
# ================================
keypoints_list, fps = extract_keypoints_yolo(VIDEO_PATH)

if not keypoints_list:
    print("No keypoints detected — cannot classify.")
    exit()

# Build adjacency matrix
num_nodes = 17
self_link = [(i, i) for i in range(num_nodes)]
neighbor_link = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (5, 11), (6, 12)
]
edge = self_link + neighbor_link
A = np.zeros((num_nodes, num_nodes))
for i, j in edge:
    A[i, j] = 1
    A[j, i] = 1
A = torch.from_numpy(A).float().to(DEVICE)

# Load SGT model
model = SGT_Model(num_classes=2, in_channels=2, num_nodes=num_nodes).to(DEVICE)
model.load_state_dict(torch.load(SGT_MODEL_PATH, map_location=DEVICE))
model.eval()

# Convert list to numpy array
keypoints_array = np.array(keypoints_list, dtype=np.float32)
num_total_frames = keypoints_array.shape[0]

print(f"\nProcessed {num_total_frames} frames.")
print("Running classification...\n")

for start_frame in range(0, num_total_frames, STRIDE):
    end_frame = start_frame + MAX_FRAMES
    chunk = keypoints_array[start_frame:end_frame]

    if len(chunk) < fps:
        continue

    # (T, V, C) → (C, T, V)
    data = np.transpose(chunk, (2, 0, 1))

    # Pad/truncate
    c, t, v = data.shape
    if t < MAX_FRAMES:
        pad = np.zeros((c, MAX_FRAMES - t, v), dtype=np.float32)
        data = np.concatenate((data, pad), axis=1)
    else:
        data = data[:, :MAX_FRAMES, :]

    data = np.expand_dims(data, axis=0)
    input_tensor = torch.from_numpy(data).float().to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor, A)
        probs = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)

        prediction = CLASS_NAMES[pred_idx.item()]
        confidence = conf.item() * 100

    print(f"{start_frame/fps:05.1f}s → {prediction} ({confidence:.1f}%)")
