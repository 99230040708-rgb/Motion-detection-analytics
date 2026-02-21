import streamlit as st
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Video Analytics", layout="wide")

st.title("🎯 Video Motion Analytics Dashboard")

uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

if uploaded_video is not None:

    # Create folders
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    video_path = "input/input.mp4"

    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.success("✅ Video uploaded successfully")

    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        st.error("Unable to read video.")
        st.stop()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    results = []
    heatmap_accumulator = None

    preview = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Motion detection
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_count = 0

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                motion_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

                if heatmap_accumulator is None:
                    heatmap_accumulator = np.zeros(gray.shape, dtype=np.float32)

                heatmap_accumulator[y:y+h, x:x+w] += 1

        results.append([frame_count, motion_count])
        prev_gray = gray

        preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Save CSV
    df = pd.DataFrame(results, columns=["Frame", "Motion_Count"])
    df.to_csv("output/report.csv", index=False)

    st.success("🎉 Processing Completed")

    # =========================
    # 📊 Analytics Dashboard
    # =========================
    st.subheader("📊 Motion Analytics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Frames", frame_count)
    col2.metric("Total Motion Events", int(df["Motion_Count"].sum()))
    col3.metric("Avg Motion / Frame", round(df["Motion_Count"].mean(), 2))

    st.line_chart(df.set_index("Frame")["Motion_Count"])
    st.bar_chart(df.set_index("Frame")["Motion_Count"])

    # =========================
    # 🔥 Compact Motion Heatmap
    # =========================
    st.subheader("🔥 Motion Heatmap")

    if heatmap_accumulator is not None:
        heatmap = cv2.normalize(
            heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # 👇 create centered narrow column to prevent stretching
        left, center, right = st.columns([1, 1.5, 1])

        with center:
            fig, ax = plt.subplots(figsize=(4,4))
            im = ax.imshow(heatmap, cmap="hot")
            ax.axis("off")

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Motion Intensity", rotation=270, labelpad=12)

            st.pyplot(fig, use_container_width=False)

    else:
        st.info("No motion detected to generate heatmap.")

    st.markdown("""
### 🔍 Heatmap Color Guide
- **Black / Dark** → No motion  
- **Red** → Low activity  
- **Orange** → Light movement  
- **Yellow** → Moderate activity  
- **Bright Yellow / White** → Highest activity zones  
""")

    # =========================
    # ⬇ Download Report
    # =========================
    with open("output/report.csv","rb") as f:
        st.download_button("⬇ Download Report", f, file_name="motion_report.csv")