import cv2
import numpy as np
import pandas as pd

video_path = "input/input.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video not opened")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

expected_interval = 1.0 / fps

prev_gray = None
frame_index = 0
results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    label = "Normal"
    color = (0,255,0)

    if prev_gray is not None:

        # Motion difference
        diff = cv2.absdiff(prev_gray, gray)
        motion = np.mean(diff)

        # Blur detection
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Optical flow detection
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        flow_motion = np.mean(mag)

        # Classification logic
        if blur_score < 20:
            label = "Blur/Corrupted"
            color = (0,0,255)

        elif flow_motion > 12:
            label = "Frame Merge"
            color = (255,0,0)

        elif motion < 2:
            label = "Frame Drop"
            color = (0,255,255)

    # Overlay label
    cv2.putText(frame, label, (40,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    results.append([frame_index, label])

    cv2.imshow("Anomaly Detection", frame)

    prev_gray = gray
    frame_index += 1

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Save CSV
df = pd.DataFrame(results, columns=["Frame","Label"])
df.to_csv("output/report.csv", index=False)

print("Done")