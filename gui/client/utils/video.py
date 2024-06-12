import time
import threading
import numpy as np
import cv2
import os
import streamlit as st
import requests
from queue import Queue
import concurrent.futures
from utils.model import load_yolo_model, load_keras_model
from datetime import datetime

pose_model = load_yolo_model('models/yolov8n-pose.pt')
lstm_model = load_keras_model('models/lstm_model.keras')
object_model = load_yolo_model('models/best_last.pt')

sequence_length = 100
sequence = []

saving = False

frame_queue = Queue(maxsize=10)
result_queue = Queue(maxsize=10)
flag_queue = Queue(maxsize=10)

banner_detected_time = None
trash_detected_time = None

save_dir = "videos/"

def preprocess_frame(frame):
    results = pose_model(frame, conf = 0.6)
    if not results or not results[0].keypoints or len(results[0].keypoints.xy[0]) == 0:
        keypoints_flat = np.zeros(34)
    else:
        keypoints = results[0].keypoints.xy
        keypoints = keypoints[0].cpu().numpy()
        keypoints_flat = keypoints.flatten()
    return keypoints_flat

def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

def predict_with_model1(frame):
    person_boxes = []
    action = None
    frame_processed = preprocess_frame(frame)
    sequence.append(frame_processed)
    if len(sequence) > sequence_length:
        sequence.pop(0)
    if len(sequence) == sequence_length:
        input_sequence = np.expand_dims(np.array(sequence), axis=0)
        prediction = lstm_model.predict(input_sequence)
        predicted_label = np.argmax(prediction, axis=1).flatten()[0]
        if predicted_label == 2:
            action = 'Trash'
        elif predicted_label == 3:
            action = 'Banner'
        elif predicted_label == 4:
            action = "Smoke"
        elif predicted_label == 1:
            action = 'default'
        else:
            action = 'default'
    results = pose_model(frame, conf=0.6)
    for result in results:
        if result.boxes is not None:
            for box in result.boxes.xyxy:
                person_boxes.append(box.tolist())
    return frame, person_boxes, action

def predict_with_model2(frame, person_boxes, person_flags, action):
    results = object_model(frame)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = object_model.names[cls]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if cls == 3:
                for i, person_box in enumerate(person_boxes):
                    if is_overlapping([x1, y1, x2, y2], person_box) and action == 'Trash':
                        person_flags[i] = "holding_trash"
                        break
            elif cls == 0:
                for i, person_box in enumerate(person_boxes):
                    if is_overlapping([x1, y1, x2, y2], person_box) and action == 'Banner':
                        person_flags[i] = "near_banner"
                        break
                    
    for i in range(len(person_flags)):
        if person_flags[i] in ['holding_trash', 'near_banner']:
            continue
        if action == 'Trash':
            person_flags[i] = 'not_holding_trash'
        elif action == 'Banner':
            person_flags[i] = 'not_near_banner'
        elif action == 'default':
            person_flags[i] = 'general'
    for i, (x1, y1, x2, y2) in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if person_flags[i] == "holding_trash":
            color = (0, 0, 255)
            cv2.putText(frame, "trash_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif person_flags[i] == "near_banner":
            color = (0, 0, 255)
            cv2.putText(frame, "banner_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif person_flags[i] == "not_holding_trash":
            color = (0, 255, 255)
            cv2.putText(frame, "not_holding_trash_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif person_flags[i] == "not_near_banner":
            color = (0, 255, 255)
            cv2.putText(frame, "not_near_banner_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame, person_flags

video_path = 'videos/20240612_163507.mp4'

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
    stop_event.set()

def send_log(section, action):
    now = datetime.now()
    log_date = now.strftime("%Y-%m-%d")
    log_time = now.strftime("%H:%M:%S")

    log_data = {
        'date': log_date,
        'time': log_time,
        'section': section,
        'action': action
    }

    requests.post(st.secrets.address.address + '/log/upload', json=log_data)

def frame_processor(stop_event):
    global saving
    global banner_detected_time
    global output_video_writer
    global trash_detected_time

    output_video_writer = None
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

                if "near_banner" in person_flags and not saving:
                    flag_queue.put("banner_person_detected")
                    banner_detected_time = datetime.now()
                    saving = True
                    send_log("A", "현수막")
                elif "holding_trash" in person_flags and not saving:
                    flag_queue.put("trash_person_detected")
                    trash_detected_time = datetime.now()
                    saving = True
                    send_log("A", "무단 투기")
                if saving:
                    elapsed_time_banner = (datetime.now() - banner_detected_time).total_seconds() if banner_detected_time else float('inf')
                    elapsed_time_trash = (datetime.now() - trash_detected_time).total_seconds() if trash_detected_time else float('inf')
                    if elapsed_time_banner <= 30 or elapsed_time_trash <= 30:
                        if output_video_writer is None:
                            if banner_detected_time and elapsed_time_banner <= 30:
                                video_file = os.path.join(save_dir, f"banner_{banner_detected_time.strftime('%Y-%m-%d_%H:%M:%S')}.avi")
                            elif trash_detected_time and elapsed_time_trash <= 30:
                                video_file = os.path.join(save_dir, f"trash_{trash_detected_time.strftime('%Y-%m-%d_%H:%M:%S')}.avi")
                            output_video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))
                        output_video_writer.write(blended_frame)    
                    else:
                        saving = False
                        banner_detected_time = None
                        trash_detected_time = None
                        if output_video_writer is not None:
                            output_video_writer.release()
                            output_video_writer = None
                else:
                    if output_video_writer is not None:
                        output_video_writer.release()
                        output_video_writer = None
                if result_queue.full():
                    result_queue.get()
                result_queue.put(blended_frame)
        stop_event.set()



def process():
    global saving
    stop_event = threading.Event()
    reader_thread = threading.Thread(target=frame_reader, args=(stop_event,))
    processor_thread = threading.Thread(target=frame_processor, args=(stop_event,))
    reader_thread.start()
    processor_thread.start()
    frame_count = 0
    while not stop_event.is_set():
        if not result_queue.empty():
            blended_frame = result_queue.get()
            blended_frame_rgb = cv2.cvtColor(blended_frame, cv2.COLOR_BGR2RGB)
            st.image(blended_frame_rgb)
            frame_count += 1
        else:
            time.sleep(0.01)

    stop_event.set()
    reader_thread.join()
    processor_thread.join()