import threading
import time
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from queue import Queue
import concurrent.futures

# Load models
pose_model = YOLO('yolov8n-pose.pt')
lstm_model = load_model('lstm_model.keras')
object_model = YOLO('best.pt')

# Sequence length setting
sequence_length = 36
sequence = []

# Frame preprocessing function
def preprocess_frame(frame):
    results = pose_model(frame, conf = 0.6)
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

# Frame prediction function with model 1
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
        predicted_label = np.argmax(prediction, axis=1).flatten()[0]  # Use softmax for multi-class prediction
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
        cv2.putText(frame, f'Action: {action}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Extract coordinates of bounding boxes of human objects (no color setting)
    results = pose_model(frame, conf=0.6)
    for result in results:
        if result.boxes is not None:
            for box in result.boxes.xyxy:
                person_boxes.append(box.tolist())
    return frame, person_boxes, action

# Frame prediction function with model 2
def predict_with_model2(frame, person_boxes, person_flags, action):
    results = object_model(frame)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)  # Class label
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Draw rectangle around detected object
            label = object_model.names[cls]
            color = (0, 255, 0)  # Green for other objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if cls == 3:  # garbage_bag
                for i, person_box in enumerate(person_boxes):
                    if is_overlapping([x1, y1, x2, y2], person_box) and action == 'Trash':
                        person_flags[i] = "holding_trash"
                        break
            elif cls == 0:  # banner
                for i, person_box in enumerate(person_boxes):
                    if is_overlapping([x1, y1, x2, y2], person_box) and action == 'Banner':
                        person_flags[i] = "near_banner"
                        break

    # Set person flags to yellow if action is 'Trash' or 'Banner' and not holding trash or near banner
    for i in range(len(person_flags)):
        if person_flags[i] in ['holding_trash', 'near_banner']:
            continue
        if action == 'Trash':
            person_flags[i] = 'not_holding_trash'
        elif action == 'Banner':
            person_flags[i] = 'not_near_banner'
        elif action == 'default':
            person_flags[i] = 'general'

    # Change border color according to the state of human objects
    for i, (x1, y1, x2, y2) in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if person_flags[i] == "holding_trash":
            color = (0, 0, 255)  # Red
            cv2.putText(frame, "trash_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif person_flags[i] == "near_banner":
            color = (0, 0, 255)  # Red
            cv2.putText(frame, "banner_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif person_flags[i] == "not_holding_trash":
            color = (0, 255, 255)  # Yellow for not holding trash
            cv2.putText(frame, "not_holding_trash_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif person_flags[i] == "not_near_banner":
            color = (0, 255, 255)  # Yellow for not near banner
            cv2.putText(frame, "not_near_banner_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame, person_flags

# Video path setting
video_path = '/home/john/Downloads/20240612_163507.mp4'

# Queue setting
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
                alpha = 0
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

while True:
    if not result_queue.empty():
        blended_frame = result_queue.get()
        cv2.imshow("YOLOv8 Inference", blended_frame)
        frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()
        break

cv2.destroyAllWindows()
