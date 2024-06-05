import streamlit as st
import requests
import time
import base64
import cv2

from http import HTTPStatus
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
from utils.yoloProcess import YOLOProcessor
from datetime import datetime

class Page:
    def __init__(self):
        if 'page' not in st.session_state:
            st.session_state.page = 'sign_in'

    def set_page(self, page_name):
        st.session_state.page = page_name
        st.experimental_rerun()

    def sign_in_page(self):
        st.set_page_config(page_title='Park Management', page_icon='images/favicon.ico', layout='centered')

        self.sign_in_layout = st.columns(1)
        with self.sign_in_layout[0]:
            title_cols = st.columns([1, 1, 3.3])
            with title_cols[2]:
                st.header('로그인')

            hide_menu_style = '<style>[data-testid="InputInstructions"] { display:None } </style>'
            st.markdown(hide_menu_style, unsafe_allow_html=True)
            input_id = st.text_input(label='아이디', max_chars=20, placeholder='아이디를 입력해주세요.', autocomplete=None)
            input_password = st.text_input(label='비밀번호', max_chars=20, type='password', placeholder='비밀번호를 입력해주세요.', autocomplete=None)

            cols = st.columns([6.1, 1, 1.2])
            with cols[1]:
                st.button(label='로그인', on_click=lambda: self.sign_in(input_id, input_password))
            with cols[2]:
                if st.button(label='회원가입'):
                    self.set_page('sign_up')

    def sign_up_page(self):
        st.set_page_config(page_title='Park Management', page_icon='images/favicon.ico', layout='centered')

        self.sign_up_layout = st.columns(1)
        with self.sign_up_layout[0]:
            title_cols = st.columns([1, 1, 3.8])
            with title_cols[2]:
                st.header('회원가입')

            hide_menu_style = '<style>[data-testid="InputInstructions"] { display:None } </style>'
            st.markdown(hide_menu_style, unsafe_allow_html=True)
            user_name = st.text_input(label='이름', max_chars=10, placeholder='이름을 입력해주세요.', autocomplete='')
            user_id = st.text_input(label='아이디', max_chars=20, placeholder='아이디를 입력해주세요.', autocomplete='')
            user_password = st.text_input(label='비밀번호', type='password', max_chars=20, placeholder='비밀번호를 입력해주세요.', autocomplete='')

            cols = st.columns([5.9, 1.2, 1.2])
            with cols[1]:
                if st.button(label='뒤로가기'):
                    self.set_page('sign_in')
            with cols[2]:
                st.button(label='가입하기', on_click=lambda: self.sign_up(user_name, user_id, user_password))

    def sign_in(self, user_id, user_password):
        with self.sign_in_layout[0]:
            if len(user_id) == 0:
                st.write(':red[아이디를 입력해주세요.]')
            elif len(user_password) == 0:
                st.write(':red[비밀번호를 입력해주세요.]')
            else:
                user_data = { 'user_id': user_id, 'user_password': user_password }
                response = requests.post(st.secrets.address.address + '/auth/signin', json=user_data)

                data = response.json()
                name = data['user_name']

                if response.status_code == HTTPStatus.OK.value:
                    st.toast(f'{name}님 환영합니다.')
                    time.sleep(1)
                    self.set_page('admin')
                else:
                    st.toast('로그인 실패')
                    st.write(':red[아이디 혹은 비밀번호가 틀렸습니다.]')

    def sign_up(self, user_name, user_id, user_password):
        with self.sign_up_layout[0]:
            if len(user_name) == 0:
                st.write(':red[이름을 입력해주세요.]')
            elif len(user_id) == 0:
                st.write(':red[아이디를 입력해주세요.]')
            elif len(user_password) == 0:
                st.write(':red[비밀번호를 입력해주세요.]')
            else:
                user_data = {'user_name': user_name, 'user_id': user_id, 'user_password': user_password}
                response = requests.post(st.secrets.address.address + '/auth/signup', json=user_data)
                
                if response.status_code == HTTPStatus.NOT_FOUND.value:
                    st.toast('회원가입 성공')
                    time.sleep(1)
                    self.set_page('sign_in')
                else:
                    st.toast('이미 가입한 적이 있습니다.')

    def admin_page(self):
        st.set_page_config(page_title='Park Management', page_icon='images/favicon.ico', layout='wide', initial_sidebar_state='expanded')
        st.title('Park Management System')
        self.pages = { 'CCTV': self.cctv_page, 'LOG': self.log_page }
        page = st.sidebar.selectbox('Select Page', list(self.pages.keys()))
        self.pages[page]()
    
    def cctv_page(self):
        st.subheader("CCTV")
        tabs = st.tabs(['A구역','CCTV 구역 추가'])
        with tabs[0]:
            st.header('A구역')
            cctv1_path = "videos/action1.mp4"
            self.cctv1(cctv1_path)
        with tabs[1]:
            st.header('CCTV 추가')
            self.upload_video()

    @st.cache_resource
    def load_model(self, model_path):
        model = YOLO(model_path)
        return model
    
    def cctv1(self, video_path):
        banner_detected = False
        banner_start_time = None
        banner_end_time = None
        model = self.load_model('models/best.pt')
        cap = cv2.VideoCapture(video_path)
        frame_window = st.image([])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                if model.names[int(cls)] == 'banner_per':
                    if not banner_detected:
                        banner_detected = True
                        banner_start_time = datetime.now()
            if banner_detected:
                banner_end_time = datetime.now()
                st.info(f"Banner detected from {banner_start_time.strftime('%Y-%m-%d %H:%M:%S')} to {banner_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                banner_detected = False
                # if banner_detected:
                #     banner_duration = banner_end_time - banner_start_time
                #     minutes, seconds = divmod(banner_duration.total_seconds(), 60)
                #     st.info(f"Banner detected for {int(minutes)} minutes and {int(seconds)} seconds")
                #     banner_detected = False  # 다음 인식을 위해 초기화
            frame_window.image(frame, channels='BGR')
        cap.release()
        
    def upload_video(self):
        st.subheader('CCTV ADD')
        uploaded_file = st.file_uploader('Select CCTV to add', type=['mp4'])
        if uploaded_file is not None:
            self.play_video(uploaded_file)
            
    def play_video(self, uploaded_file):
        st.subheader('CCTV')
        video_bytes = uploaded_file.read()
        st.video(video_bytes)
        
    def log_page(self):
        sidebar_cols = st.sidebar.columns(2)
        with sidebar_cols[0]:
            st.date_input('시작일',)
        with sidebar_cols[1]:
            st.date_input('종료일')
        st.sidebar.button('적용', use_container_width=True)

    def run(self):
        if st.session_state.page == 'sign_in':
            self.sign_in_page()
        if st.session_state.page == 'sign_up':
            self.sign_up_page()
        if st.session_state.page == 'admin':
            self.admin_page()

if __name__ == '__main__':
    page = Page()
    page.run()