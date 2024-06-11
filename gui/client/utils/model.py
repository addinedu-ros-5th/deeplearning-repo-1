import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model

@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

@st.cache_resource
def load_keras_model(model_path):
    model = load_model(model_path)
    return model