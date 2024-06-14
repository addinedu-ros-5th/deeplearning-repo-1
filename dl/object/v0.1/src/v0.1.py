import cv2
from ultralytics import YOLO
import concurrent.futures

# 첫 번째 YOLO 모델 객체 생성
model1 = YOLO('/home/john/dev_ws/yolo/runs/detect/train97/weights/best.pt')

# 두 번째 YOLO 모델 객체 생성
model2 = YOLO('yolov8n-pose.pt')

video_path = '/home/john/Downloads/20240604_201630.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 클래스 아이디를 클래스 이름으로 매핑하는 딕셔너리
class_id_to_name = {
    0.0: "garbage_bag",
}

# 첫 번째 모델 실행 함수
def run_model1(frame):
    current_frame = frame.copy()
    trash_bag_coords = []

    results1 = model1(current_frame)
    for result in results1:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = box.cls[0].item()
            class_name = class_id_to_name.get(class_id, "Unknown")
            label = f'{class_name}: {confidence:.2f}'
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(current_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if class_name == "garbage_bag":
                trash_bag_coords.append((x1, y1, x2, y2))

    return current_frame, trash_bag_coords

# 두 번째 모델 실행 함수
def run_model2(frame):
    current_frame = frame.copy()
    person_boxes = []

    results2 = model2(current_frame)
    for result in results2:
        boxes = result.boxes.xyxy if result.boxes else None
        if boxes is not None:
            person_boxes.extend(boxes.tolist())

    return current_frame, person_boxes

def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 두 박스가 겹치거나 침범하는 경우를 감지
    return (x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min)

# 이전 프레임의 좌표 저장용 변수
prev_person_boxes = []
prev_trash_bag_coords = []
person_flags = []  # 각 사람의 상태를 저장하는 리스트

# 쓰레드 풀 생성
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        future1 = executor.submit(run_model1, new_frame)
        future2 = executor.submit(run_model2, new_frame)

        processed_frame1, trash_bag_coords = future1.result()
        processed_frame2, person_boxes = future2.result()

        # person_flags 리스트의 길이를 person_boxes 리스트의 길이에 맞추기
        if len(person_flags) != len(person_boxes):
            person_flags = ["general"] * len(person_boxes)

        # 각 사람에 대해 상태 확인 및 업데이트
        new_person_flags = ["general"] * len(person_boxes)
        for i, person_box in enumerate(person_boxes):
            is_holding_trash = any(boxes_overlap(person_box, trash_box) for trash_box in trash_bag_coords)
            if is_holding_trash:
                new_person_flags[i] = "holding_trash"
            else:
                if person_flags[i] == "holding_trash":
                    new_person_flags[i] = "dropped_trash"
                elif person_flags[i] == "dropped_trash":
                    new_person_flags[i] = "dropped_trash"

        # 업데이트된 상태를 person_flags에 적용
        person_flags = new_person_flags

                # 각 사람에 대해 색상으로 구분하여 테두리 그리기
        for i, (x1, y1, x2, y2) in enumerate(person_boxes):
            if person_flags[i] == "holding_trash":
                color = (0, 255, 255)  # 노란색
                print("들고가는중")
            elif person_flags[i] == "dropped_trash":
                color = (0, 0, 255)  # 빨간색
                print("쓰레기 투기!!!")
                cv2.putText(processed_frame2, "trash_person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                color = (0, 255, 0)  # 초록색
                print("일반인")
            cv2.rectangle(processed_frame2, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


        # 현재 프레임의 사람 객체와 쓰레기 봉투 좌표를 이전 프레임의 변수에 저장
        prev_person_boxes = person_boxes
        prev_trash_bag_coords = trash_bag_coords

        alpha = 0.5
        combined_frame = cv2.addWeighted(processed_frame1, alpha, processed_frame2, 1 - alpha, 0)

        cv2.imshow('Combined Results', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Both models have completed their tasks.")
