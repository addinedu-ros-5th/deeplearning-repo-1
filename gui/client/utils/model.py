import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model