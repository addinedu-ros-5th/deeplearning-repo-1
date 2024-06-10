import os
import cv2
import time
import requests
import pandas as pd
import streamlit as st

from http import HTTPStatus
from utils.file import find_file
from utils.model import load_model
from datetime import datetime, timedelta

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

            hide_style = '<style> [data-testid="InputInstructions"] { display:None } </style>'
            st.markdown(hide_style, unsafe_allow_html=True)
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

            hide_style = '<style> [data-testid="InputInstructions"] { display:None } </style>'
            st.markdown(hide_style, unsafe_allow_html=True)
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
        self.pages = { 'RECORD': self.record_page, 'CCTV': self.cctv_page }
        page = st.sidebar.selectbox('Select Page', list(self.pages.keys()))
        self.pages[page]()

    def cctv_page(self):
        st.subheader("CCTV")
        sidebar_cols = st.sidebar.columns(2)
        with sidebar_cols[0]:
            st.date_input('시작일', disabled=True)
        with sidebar_cols[1]:
            st.date_input('종료일', disabled=True)
        st.sidebar.multiselect('필터', ['A', '흡연', '투기', '음주'], default=None, placeholder='선택해주세요.', disabled=True)
        st.sidebar.button('적용', use_container_width=True, disabled=True)

        tabs = st.tabs(['A구역','CCTV 구역 추가'])
        with tabs[0]:
            st.header('A구역')
            cctv1_path = "videos/2024-01-01_123020.mp4"
            self.cctv1(cctv1_path)
        with tabs[1]:
            st.header('CCTV 추가')
            self.upload_video()

    def record_page(self):
        st.subheader('RECORD')
        sidebar_cols = st.sidebar.columns(2)
        with sidebar_cols[0]:
            start_date = st.date_input('시작일')
        with sidebar_cols[1]:
            end_date = st.date_input('종료일')
        section = st.sidebar.multiselect('구역', ['A'], default=None, placeholder='선택해주세요.')
        action = st.sidebar.multiselect('행위', ['흡연', '투기', '음주'], default=None, placeholder='선택해주세요.')
        st.sidebar.button('적용', use_container_width=True, on_click=lambda: self.apply(start_date, end_date, section, action))

        layouts = st.columns([4, 6])

        with layouts[0]:
            st.write('LOG')
            self.log = st.container(border=True)
            with self.log:
                st.empty()
        with layouts[1]:
            st.write('VIDEO')
            self.video = st.container(border=True)
            with self.video:
                st.empty()

        with self.log:
            if 'df' in st.session_state:
                event = st.dataframe(st.session_state.df, selection_mode='single-row', key='log', use_container_width=True, on_select='rerun')
                with self.video:
                    row = event.selection['rows']
                    if len(row) > 0:
                        file_name = st.session_state.df.index[row[0]]
                        st.video(find_file(file_name)[0])

    def apply(self, start_date, end_date, sections=None, actions=None):
        if 'df' in st.session_state:
            st.session_state.pop('df')

        data = {'start_date': str(start_date), 'end_date': str(end_date), 'sections': sections , 'actions': actions}

        response = requests.post(st.secrets.address.address + '/log/download', json=data)

        log_data = response.json()

        log_df = pd.DataFrame(log_data, columns=['날짜', '구역', '시작시간', '종료시간', '행위'])
        log_df = log_df.set_index(keys='날짜')

        st.session_state.df = log_df
    
    def cctv1(self, video_path):
        banner_detected = False
        banner_start_time = None
        banner_end_time = None

        trash_detected = False
        trash_start_time = None
        trash_end_time = None

        model1 = load_model('models/best.pt')
        model2 = load_model('models/best (1).pt')
        model3 = load_model('models/yolov8n-pose.pt')

        cap = cv2.VideoCapture(video_path)
        frame_window = st.image([])
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        saving = False
        frame_count = 0

        save_dir = "videos/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            fixed_size = (1280, 720)
            frame = cv2.resize(frame, fixed_size)
            frame_height, frame_width = frame.shape[:2]

            results = model1(frame)[0]
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{model1.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            results = model2(frame)[0]
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{model2.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            results = model3(frame)[0]
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{model3.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            results = model3(frame)[0]
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{model3.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                if model1.names[int(cls)] == 'banner_per':
                    if not banner_detected:
                        banner_detected = True
                        banner_start_time = datetime.now()

                        video_file = os.path.join(save_dir, f"detected_banner_{banner_start_time.strftime('%Y-%m-%d_%H%M%S')}.avi")
                        out = cv2.VideoWriter(video_file, fourcc, 20.0, (frame_width, frame_height))
                        if not out.isOpened():
                            st.error("Error: Failed to open video writer")
                        else:
                            st.info(f"Video recording started: {video_file}")
                        saving = True
                        frame_count = 0

                if model2.names[int(cls)] == 'dropped_trash':
                    if not trash_detected:
                        trash_detected = True
                        trash_start_time = datetime.now()

                        video_file = os.path.join(save_dir, f"detected_trash_{trash_start_time.strftime('%Y-%m-%d_%H%M%S')}.avi")
                        out = cv2.VideoWriter(video_file, fourcc, 20.0, (frame_width, frame_height))
                        if not out.isOpened():
                            st.error("Error: Failed to open video writer")
                        else:
                            st.info(f"Video recording started: {video_file}")
                        saving = True
                        frame_count = 0

            if banner_detected:
                banner_end_time = datetime.now()
                if saving:
                    out.write(frame)
                    frame_count += 1
                if banner_end_time - banner_start_time > timedelta(seconds=30):
                    saving = False
                    if out:
                        out.release()
                        st.info(f"Video recording stopped: {video_file}")
                        st.info(f"Total frames written: {frame_count}")
                    st.info(f"Banner detected from {banner_start_time.strftime('%H:%M:%S')} to {banner_end_time.strftime('%H:%M:%S')}")
                    banner_detected = False

            if trash_detected:
                trash_end_time = datetime.now()
                if saving:
                    out.write(frame)
                    frame_count += 1
                if trash_end_time - trash_start_time > timedelta(seconds=30):
                    saving = False
                    if out:
                        out.release()
                        st.info(f"Video recording stopped: {video_file}")
                        st.info(f"Total frames written: {frame_count}")
                    st.info(f"trash detected from {trash_start_time.strftime('%H:%M:%S')} to {trash_end_time.strftime('%H:%M:%S')}")
                    trash_detected = False

            frame_window.image(frame, channels='BGR')

        cap.release()
        if out is not None:
            out.release()  
        
    def upload_video(self):
        st.subheader('CCTV ADD')
        uploaded_file = st.file_uploader('Select CCTV to add', type=['mp4'])
        if uploaded_file is not None:
            self.play_video(uploaded_file)
            
    def play_video(self, uploaded_file):
        st.subheader('CCTV')
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

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