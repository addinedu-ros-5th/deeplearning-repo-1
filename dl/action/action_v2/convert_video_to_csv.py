from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np

# YOLOv8 모델 로드
model = YOLO('yolov8n-pose.pt')

# 비디오 경로 설정
video_path = '/home/john/Downloads/20240612_163507.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not cap.isOpened():
    print("비디오를 열 수 없습니다.")
    exit()

# 데이터 저장을 위한 리스트 초기화
data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    
    for result in results:
        keypoints = result.keypoints.xy if result.keypoints else None
        
        if keypoints is not None and len(keypoints[0]) >= 8:
            for person_keypoints in keypoints:
                person_keypoints = person_keypoints.cpu().numpy()  # 텐서를 NumPy 배열로 변환
                
                if len(person_keypoints) > 5:
                    left_hand_y = person_keypoints[9][1]
                    right_hand_y = person_keypoints[10][1]
                    left_leg_y = person_keypoints[13][1]
                    right_leg_y = person_keypoints[14][1]
                    left_shoulder_y = person_keypoints[5][1]
                    right_shoulder_y = person_keypoints[6][1]

                    # 라벨링 조건에 따라 라벨 부여
                    if left_hand_y > left_leg_y or right_hand_y > right_leg_y:
                        label = 2  # 쓰레기 버리는 사람
                    elif left_hand_y < left_shoulder_y and right_hand_y < right_shoulder_y:
                        label = 3  # 현수막 묶는 사람
                    # elif (left_hand_y < left_shoulder_y) != (right_hand_y < right_shoulder_y):
                    #     label = 4 # 흡연 하는 사람
                    else:
                        label = 1  # 디폴트

                    keypoints_flat = person_keypoints.flatten()
                    data.append(np.append(keypoints_flat, label))
        else:
            # 키 포인트가 8개 미만인 경우 모든 값을 0으로 채웁니다.
            keypoints_flat = np.zeros(34)  # 키 포인트가 17개이므로 34개의 0으로 채웁니다.
            data.append(np.append(keypoints_flat, 0))  # 라벨 0 (사람 인식 안됨)

cap.release()

# 데이터프레임 생성
columns = [f'x{i}' for i in range(17)] + [f'y{i}' for i in range(17)] + ['label']
df = pd.DataFrame(data, columns=columns)

# 데이터 정규화
min_vals = df.iloc[:, :-1].min()
max_vals = df.iloc[:, :-1].max()
df.iloc[:, :-1] = (df.iloc[:, :-1] - min_vals) / (max_vals - min_vals)

# CSV 파일로 저장
df.to_csv('deeplearning_demo.csv', index=False)
print("정규화된 데이터가 CSV 파일로 저장되었습니다.")
