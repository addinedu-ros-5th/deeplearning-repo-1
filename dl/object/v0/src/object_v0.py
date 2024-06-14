import cv2
from ultralytics import YOLO

# 모델 불러오기
model = YOLO('/home/john/dev_ws/yolo/runs/detect/train87/weights/best.pt')

# 입력 동영상 경로 설정
input_video_path = "/home/john/Downloads/1000010319.mp4"

# 동영상 캡처 객체 생성
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# 프레임 단위로 동영상 처리 및 재생
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 모델 예측
    results = model(frame, conf = 0.6) # 60퍼센트 이상일 때 출력

    # 결과를 프레임에 적용
    processed_frame = results[0].plot()

    # 프레임을 화면에 표시
    cv2.imshow('Real-time Object Detection', processed_frame)

    # 'q' 키를 누르면 중지
    if cv2.waitKey(int(250/fps)) & 0xFF == ord('q'):
        break

# 동영상 파일 닫기
cap.release()
cv2.destroyAllWindows()

print("동영상 재생이 완료되었습니다.")
