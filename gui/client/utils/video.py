import os
import cv2
import streamlit as st

from utils.model import load_yolo_model
from datetime import datetime, timedelta

def process(video_path):
    banner_detected = False
    banner_start_time = None
    banner_end_time = None

    trash_detected = False
    trash_start_time = None
    trash_end_time = None

    model1 = load_yolo_model('models/best.pt')
    model2 = load_yolo_model('models/best (1).pt')
    model3 = load_yolo_model('models/yolov8n-pose.pt')

    cap = cv2.VideoCapture(video_path)
    frame_window = st.image([])
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    saving = False
    frame_count = 0

    save_dir = "videos/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        fixed_size = (1280, 720)
        frame = cv2.resize(frame, fixed_size)
        frame_height, frame_width = frame.shape[:2]

        results = model1(frame)[0]
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model1.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        results = model2(frame)[0]
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model2.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        results = model3(frame)[0]
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model3.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        results = model3(frame)[0]
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model3.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if model1.names[int(cls)] == 'banner_per':
                if not banner_detected:
                    banner_detected = True
                    banner_start_time = datetime.now()

                    video_file = os.path.join(save_dir, f"detected_banner_{banner_start_time.strftime('%Y-%m-%d_%H%M%S')}.avi")
                    out = cv2.VideoWriter(video_file, fourcc, 20.0, (frame_width, frame_height))
                    if not out.isOpened():
                        st.error("Error: Failed to open video writer")
                    else:
                        st.info(f"Video recording started: {video_file}")
                    saving = True
                    frame_count = 0

            if model2.names[int(cls)] == 'dropped_trash':
                if not trash_detected:
                    trash_detected = True
                    trash_start_time = datetime.now()

                    video_file = os.path.join(save_dir, f"detected_trash_{trash_start_time.strftime('%Y-%m-%d_%H%M%S')}.avi")
                    out = cv2.VideoWriter(video_file, fourcc, 20.0, (frame_width, frame_height))
                    if not out.isOpened():
                        st.error("Error: Failed to open video writer")
                    else:
                        st.info(f"Video recording started: {video_file}")
                    saving = True
                    frame_count = 0

        if banner_detected:
            banner_end_time = datetime.now()
            if saving:
                out.write(frame)
                frame_count += 1
            if banner_end_time - banner_start_time > timedelta(seconds=30):
                saving = False
                if out:
                    out.release()
                    st.info(f"Video recording stopped: {video_file}")
                    st.info(f"Total frames written: {frame_count}")
                st.info(f"Banner detected from {banner_start_time.strftime('%H:%M:%S')} to {banner_end_time.strftime('%H:%M:%S')}")
                banner_detected = False

        if trash_detected:
            trash_end_time = datetime.now()
            if saving:
                out.write(frame)
                frame_count += 1
            if trash_end_time - trash_start_time > timedelta(seconds=30):
                saving = False
                if out:
                    out.release()
                    st.info(f"Video recording stopped: {video_file}")
                    st.info(f"Total frames written: {frame_count}")
                st.info(f"trash detected from {trash_start_time.strftime('%H:%M:%S')} to {trash_end_time.strftime('%H:%M:%S')}")
                trash_detected = False

        frame_window.image(frame, channels='BGR')

    cap.release()
    if out is not None:
        out.release()