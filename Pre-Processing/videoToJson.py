import os
import cv2
import json
from ultralytics import YOLO

# Load YOLOv11-pose model
model = YOLO("yolo11n-pose.pt")

input_dir = "Dataset/Normal"
output_dir = "Keypoints/Normal"
os.makedirs(output_dir, exist_ok=True)

def extract_keypoints(video_path, output_json):
    cap = cv2.VideoCapture(video_path)
    frame_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        keypoints = results[0].keypoints

        if keypoints is not None and len(keypoints.xy) > 0:
            # Take only the first detected person's keypoints
            kps = keypoints.xy[0].tolist()  # shape: (17, 2) → [(x, y), ...]
        else:
            # No person detected in this frame
            kps = [[0, 0] for _ in range(17)]

        frame_keypoints.append(kps)

    cap.release()

    with open(output_json, 'w') as f:
        json.dump(frame_keypoints, f)

# Loop through all Normal videos
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".mp4") and filename.startswith("N"):
        video_path = os.path.join(input_dir, filename)
        json_path = os.path.join(output_dir, filename.replace(".mp4", ".json"))
        print(f"Processing {filename}...")
        extract_keypoints(video_path, json_path)

print("✅ All keypoints extracted to Keypoints/Normal/")
