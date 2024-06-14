import cv2
import numpy as np
import os 
import mediapipe as mp
import sys

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def mediapipe_detection(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
        
    # Make detection
    results = model.process(image)
        
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
def extract_pose(results):
    # cctv 여서 zero로 처리해도 괜찮을듯 싶다. 아니면 전 데이터를 그대로 가져와야 한다.
    if results.pose_landmarks:
        if results.pose_landmarks.landmark:
            pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros([0,0])
    else:
        pose = np.zeros(66)
        
    return pose

def count_mp4_files(directory):
    count = 0
    for file in os.listdir(directory):
        if file.endswith(".mp4"):
            count += 1
    return count

def load_video(n):
    original_path = f"/home/jinsa/Downloads/5-1공원의 안내문을 훼손하는 경우/video{n}.mp4" 
    augmented_path = f"/home/jinsa/Downloads/5-1공원의 안내문을 훼손하는 경우/augmented_video{n}.mp4"
    
    # 원본 비디오 파일 로드
    cap = cv2.VideoCapture(original_path)
    
    if not cap.isOpened():
        print(f"Failed to open video file: {original_path}")
        return None

    # 비디오 파일의 총 프레임 수 확인
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 프레임 수가 300보다 낮으면 대체 비디오 파일 로드
    if frame_count < 300:
        cap.release()
        cap = cv2.VideoCapture(augmented_path)
        if not cap.isOpened():
            print(f"Failed to open augmented video file: {augmented_path}")
            return None

    return cap


directory_path = "/home/jinsa/Downloads/5-1공원의 안내문을 훼손하는 경우/"
mp4_count = count_mp4_files(directory_path)

data_path = './test_data_path/break' # path to save .npy
actions = np.array(['breaking']) # action labeling
no_sequence = 10  # Number of sequences
sequence_length = 30  # Length of each sequence

# Create data directory if it doesn't exist
os.makedirs(data_path, exist_ok=True)

# Open video capture
for n in range(1,mp4_count):
    cap = load_video(n)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for action in actions:
            action_path = os.path.join(data_path, f"video{n}")  # Modify path here
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
                       cv2.destroyAllWindows()
                       sys.exit()

    cap.release()

    cv2.destroyAllWindows()
