import os
import sys
import cv2
import random
import numpy as np
import mediapipe as mp

# Mediapipe settings
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to extract random .mp4 files
def get_random_mp4_files(directory, num_files=50):
    mp4_files = [file for file in os.listdir(directory) if file.endswith(".mp4")]
    if len(mp4_files) <= num_files:
        return mp4_files
    else:
        return random.sample(mp4_files, num_files)

# Function to load video
def load_video(base_directory, file_n):
    original_path = os.path.join(base_directory, file_n)
    cap = cv2.VideoCapture(original_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {original_path}")
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 300:
        cap.release()
        print(f"Video file {original_path} has less than 300 frames.")
        return None
    return cap

# Function for Mediapipe detection
def mediapipe_detection(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Function to extract pose
def extract_pose(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)  # 33 landmarks with 4 values each
    return pose


# Paths and settings
# (window os path)
base_directories_dict = {
    "5": r"C:\Users\jaeho\Downloads\173.공원 주요시설 및 불법행위 감시 CCTV 영상 데이터\01.데이터\1.Training\원천데이터\TS_행위(불법행위)데이터3\1.불법행위\5.시설물파손행위",
    "6": r"C:\Users\jaeho\Downloads\173.공원 주요시설 및 불법행위 감시 CCTV 영상 데이터\01.데이터\1.Training\원천데이터\TS_행위(불법행위)데이터3\1.불법행위\6.현수막부착행위",
    "7": r"C:\Users\jaeho\Downloads\173.공원 주요시설 및 불법행위 감시 CCTV 영상 데이터\01.데이터\1.Training\원천데이터\TS_행위(불법행위)데이터4\1.불법행위\7.전단지배부행위",
    "8": r"C:\Users\jaeho\Downloads\173.공원 주요시설 및 불법행위 감시 CCTV 영상 데이터\01.데이터\1.Training\원천데이터\TS_행위(불법행위)데이터4\1.불법행위\8.텐트설치행위"
} 

data_path = r'C:\Users\jaeho\Desktop\DL_video_data\skeleton'
actions = np.array(['Attaching', 'Breaking', 'Distribution', 'Setting'])
no_sequence = 3
sequence_length = 100

for idx, path in base_directories_dict.items():
    random_mp4_files = get_random_mp4_files(path)
    for n, file_n in enumerate(random_mp4_files):
        cap = load_video(path, file_n)
        if cap is None:
            continue
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            action_path = os.path.join(data_path, actions[int(idx)-5], f"video{n}")
            os.makedirs(action_path, exist_ok=True)
            for sequence in range(no_sequence):
                sequence_path = os.path.join(action_path, f"s{sequence}")
                os.makedirs(sequence_path, exist_ok=True)
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture frame")
                        break
                    image, results = mediapipe_detection(frame, pose)
                    draw_landmarks(image, results)
                    keypoints = extract_pose(results)
                    npy_path = os.path.join(sequence_path, f"{frame_num}.npy")
                    np.save(npy_path, keypoints)
                    cv2.imshow('Your video', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit()
                        
        cap.release()
        cv2.destroyAllWindows()
