import cv2
import numpy as np
import os 
import mediapipe as mp

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
    if results.pose_landmarks:
        if results.pose_landmarks.landmark:
            pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros([0,0])
    else:
        pose = np.zeros(66)
        
    return pose


data_save_path = './data/test' # path to save .npy
actions = np.array(['action1']) # action labeling
no_sequence = 10  # Number of sequences
sequence_length = 30  # Length of each sequence
video_path = './badminton_video.mp4' # path your video

os.makedirs(data_save_path, exist_ok=True)


cap = cv2.VideoCapture(video_path) 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for action in actions:
        action_path = os.path.join(data_save_path, action)
        os.makedirs(action_path, exist_ok=True)
        for sequence in range(no_sequence):
            sequence_path = os.path.join(action_path, str(sequence))
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
                    break
                
cap.release()
cv2.destroyAllWindows()