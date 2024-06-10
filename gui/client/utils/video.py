import os
import cv2
import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from utils.model import load_yolo_model
from datetime import datetime, timedelta
import threading
import time
from ultralytics import YOLO
from queue import Queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


model1 = load_yolo_model('models/best_last.pt')
model2 = load_yolo_model('models/yolov8n-pose.pt')
model3 = load_model('models/lstm_model.keras')

sequence_length = 3
sequence = []
count = 0

video_path = '/home/kjy/dev_ws/git_ws/deeplearning-repo-1/gui/client/videos/2024-01-01_123020.mp4'

def preprocess_frame(frame):
    results = model2(frame, conf=0.8)
    if not results or not results[0].keypoints or len(results[0].keypoints.xy[0]) == 0:
        keypoints_flat = np.zeros(34)
    else:
        keypoints = results[0].keypoints.xy
        keypoints = keypoints[0].cpu().numpy()  # Convert tensor to NumPy array
        keypoints_flat = keypoints.flatten()
    return keypoints_flat

# Check for overlapping bounding boxes
def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

# Apply mosaic effect to a given region in the frame
def apply_mosaic(frame, x1, y1, x2, y2, size=10):
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > frame.shape[1]: x2 = frame.shape[1]
    if y2 > frame.shape[0]: y2 = frame.shape[0]
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return
    small = cv2.resize(region, (size, size), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic

# Frame prediction function with model 1
def predict_with_model1(frame):
    global count  # Use the global count variable
    person_boxes = []
    action = None
    frame_processed = preprocess_frame(frame)
    sequence.append(frame_processed)
    if len(sequence) > sequence_length:
        sequence.pop(0)
    if len(sequence) == sequence_length:
        input_sequence = np.expand_dims(np.array(sequence), axis=0)
        prediction = model3.predict(input_sequence)
        predicted_label = np.argmax(prediction, axis=1).flatten()[0]  # Use softmax for multi-class prediction
        if predicted_label == 2:
            action = 'Trash'
        elif predicted_label == 3:
            action = 'Banner'
        elif predicted_label == 4:
            action = "Smoke"
            count += 1  # Increment the count for smoking
        elif predicted_label == 1:
            action = 'default'
        else:
            action = 'default'

    frame_with_text = frame.copy()
    cv2.putText(frame_with_text, f'Action: {action}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    results = model2(frame, conf=0.8)
    for result in results:
        if result.boxes is not None:
            for box in result.boxes.xyxy:
                person_boxes.append(box.tolist())
    for result in results:
        if result.keypoints is not None and len(result.keypoints.xy[0]) > 0:
            for keypoint in result.keypoints.xy:
                nose_x, nose_y = map(int, keypoint[0].tolist())
                x1, y1, x2, y2 = nose_x - 50, nose_y - 50, nose_x + 50, nose_y + 50
                roi = frame_with_text[y1:y2, x1:x2].copy()
                apply_mosaic(roi, 0, 0, x2-x1, y2-y1)
                frame_with_text[y1:y2, x1:x2] = roi
    return frame_with_text, person_boxes, action


def predict_with_model2(frame, person_boxes, person_flags, action):
    results = model1(frame, conf=0.8)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)  # Class label
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Draw rectangle around detected object
            label = model1.names[cls]
            color = (0, 255, 0)  # Green for other objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if cls == 3:  # garbage_bag
                for i, person_box in enumerate(person_boxes):
                    if is_overlapping([x1, y1, x2, y2], person_box) and action == 'Trash':
                        person_flags[i] = "holding_trash"
                        color = (0, 0, 255)  # Red
                        cv2.putText(frame, "trash_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        break
            elif cls == 0:  # banner
                for i, person_box in enumerate(person_boxes):
                    if is_overlapping([x1, y1, x2, y2], person_box) and action == 'Banner':
                        person_flags[i] = "banner"
                        color = (0, 255, 255)  # Yellow
                        cv2.putText(frame, "banner_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        break
    # Change border color according to the state of human objects
    for i, (x1, y1, x2, y2) in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if person_flags[i] == "holding_trash":
            color = (0, 0, 255)  # Red
            cv2.putText(frame, "trash_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif person_flags[i] == "banner":
            color = (0, 255, 255)  # Yellow for near banner
            cv2.putText(frame, "banner", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Detect noses and apply mosaic
    results = model2(frame, conf=0.8)
    for result in results:
        if result.keypoints is not None and len(result.keypoints.xy[0]) > 0:
            for keypoint in result.keypoints.xy:
                nose_x, nose_y = map(int, keypoint[0].tolist())
                x1, y1, x2, y2 = nose_x - 50, nose_y - 50, nose_x + 50, nose_y + 50
                roi = frame[y1:y2, x1:x2].copy()
                apply_mosaic(roi, 0, 0, x2-x1, y2-y1)
                frame[y1:y2, x1:x2] = roi
    return frame, person_flags


frame_queue = Queue(maxsize=10)
result_queue = Queue(maxsize=10)

def frame_reader(stop_event):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps
    while not stop_event.is_set() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)
        time.sleep(frame_time)
    cap.release()

def process(video_path):
    banner_detected = False
    banner_start_time = None
    banner_end_time = None

    trash_detected = False
    trash_start_time = None
    trash_end_time = None

    cap = cv2.VideoCapture(video_path)
    frame_window = st.image([])

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        frame_interval = 5

        if frame_count % frame_interval == 0:
            person_flags = []

    
            with ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(predict_with_model1, frame.copy())
                result1, person_boxes, action = future1.result()
                if len(person_flags) != len(person_boxes):
                    person_flags = [""] * len(person_boxes)
                future2 = executor.submit(predict_with_model2, frame.copy(), person_boxes, person_flags, action)
                result2, person_flags = future2.result()

            frame_window.image(result2, channels='BGR')

            for flag in person_flags:
                if flag == "holding_trash" and not trash_detected:
                    trash_detected = True
                    trash_start_time = datetime.now()
                    video_file = os.path.join(save_dir, f"detected_trash_{trash_start_time.strftime('%Y-%m-%d_%H%M%S')}.avi")
                    out = cv2.VideoWriter(video_file, fourcc, 20.0, (frame_width, frame_height))
                    if not out.isOpened():
                        print("Error: Failed to open video writer")
                    else:
                        print(f"Video recording started: {video_file}")
                    saving = True

                if flag == "banner" and not banner_detected:
                    banner_detected = True
                    banner_start_time = datetime.now()
                    video_file = os.path.join(save_dir, f"detected_banner_{banner_start_time.strftime('%Y-%m-%d_%H%M%S')}.avi")
                    out = cv2.VideoWriter(video_file, fourcc, 20.0, (frame_width, frame_height))
                    if not out.isOpened():
                        st.error("Error: Failed to open video writer")
                    else:
                        st.info(f"Video recording started: {video_file}")
                    saving = True

            if trash_detected:
                trash_end_time = datetime.now()
                if saving:
                    out.write(result2)
                    frame_count += 1
                if trash_end_time - trash_start_time > timedelta(seconds=30):
                    saving = False
                    if out:
                        out.release()
                        st.info(f"Video recording stopped: {video_file}")
                        st.info(f"Total frames written: {frame_count}")
                    st.info(f"Trash detected from {trash_start_time.strftime('%H:%M:%S')} to {trash_end_time.strftime('%H:%M:%S')}")
                    trash_detected = False

            if banner_detected:
                banner_end_time = datetime.now()
                if saving:
                    out.write(result2)
                    frame_count += 1
                if banner_end_time - banner_start_time > timedelta(seconds=30):
                    saving = False
                    if out:
                        out.release()
                        st.info(f"Video recording stopped: {video_file}")
                        st.info(f"Total frames written: {frame_count}")
                    st.info(f"Banner detected from {banner_start_time.strftime('%H:%M:%S')} to {banner_end_time.strftime('%H:%M:%S')}")
                    banner_detected = False

    cap.release()
    if out is not None:
        out.release()

def frame_processor(stop_event):
    person_flags = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while not stop_event.is_set():
            if not frame_queue.empty():
                frame = frame_queue.get()
                future1 = executor.submit(predict_with_model1, frame.copy())
                result1, person_boxes, action = future1.result()
                if len(person_flags) != len(person_boxes):
                    person_flags = ["general"] * len(person_boxes)
                future2 = executor.submit(predict_with_model2, frame.copy(), person_boxes, person_flags, action)
                result2, person_flags = future2.result()
                alpha = 0.5
                blended_frame = cv2.addWeighted(result1, alpha, result2, 1 - alpha, 0)
                if result_queue.full():
                    result_queue.get()
                result_queue.put(blended_frame)
# Start threads
stop_event = threading.Event()
reader_thread = threading.Thread(target=frame_reader, args=(stop_event,))
processor_thread = threading.Thread(target=frame_processor, args=(stop_event,))
reader_thread.start()
processor_thread.start()

# Set screen switching interval
switch_interval = 30  # Frame unit
frame_count = 0
# Function to blend previous and current results
prev_display_frame = None

def blend_frames(prev_frame, curr_frame, alpha):
    return cv2.addWeighted(prev_frame, alpha, curr_frame, 1 - alpha, 0)

# while True:
#     if not result_queue.empty():
#         blended_frame = result_queue.get()
#         st.image(blended_frame, channels="BGR")
#         frame_count += 1
#     if frame_count >=switch_interval:
#         stop_event.set()
#         break
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         stop_event.set()
#         break
cv2.destroyAllWindows()



