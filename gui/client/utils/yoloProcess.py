import av
import cv2
import time

from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase

class YOLOProcessor(VideoProcessorBase):
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img)[0]
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{self.model.names[int(cls)]} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        time.sleep(0.1)
        return av.VideoFrame.from_ndarray(img, format="bgr24")