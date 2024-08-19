# 행위와 객체 인식 기반 단지 내 시설물 관리 보조 시스템

## 1.프로젝트 개요

### 1.1 주제 소개
행위와 객체 인식 기반 단지 내 시설물 관리 보조 시스템

### 1.2 기술 스택
|||
|:---|:---|
|개발 환경|<img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=Linux&logoColor=white"> <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=Ubuntu&logoColor=white"> <img src="https://img.shields.io/badge/VSC-007ACC?style=for-the-badge&logo=VisualStudioCode&logoColor=white">|
|언어|<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> |
|딥러닝 및 영상처리|<img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"> <img src="https://img.shields.io/badge/Yolov8-8A2BE2?style=for-the-badge">|
|GUI|<img src="https://img.shields.io/badge/Streamlit-FF0000?style=for-the-badge&logo=streamlit&logoColor=white">
|데이터베이스|<img src="https://img.shields.io/badge/aws rds-527FFF?style=for-the-badge&logo=aws&logoColor=white"> <img src="https://img.shields.io/badge/mysql-4479A1?style=for-the-badge&logo=mysql&logoColor=white">|
|서버|<img src="https://img.shields.io/badge/flask-F6F6F6?style=for-the-badge&logo=flask&logoColor=black">|
|협업|<img src="https://img.shields.io/badge/Jira-0052CC?style=for-the-badge&logo=Jira&logoColor=white"> <img src="https://img.shields.io/badge/confluence-%23172BF4.svg?style=for-the-badge&logo=confluence&logoColor=white"> <img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white"> <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">|



### 1.3 팀원 소개
|이름|직책|담당|
|:---|:---|:---|
|강지연|팀장|GUI - CCTV Page 구현, GIT 관리|
|조성현|팀원|GUI - LOG Page 구현, 웹서버 구축|
|김요한|팀원|Action v1, Object v0, Process 구현|
|신재훈|팀원|Action v0, Object v1 구현, 발표|
|조성오|팀원|데이터 라벨링, 시나리오 구성, 협업 툴 관리|


## 2. 프로젝트 설계

### 2.1 소프트웨어 요구사항
![image](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/163802905/bd7e6b60-2a8c-4a6b-b194-5766ef4a2d67)


### 2.2 시스템구성도
![image](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/86091697/8f03b198-dee2-423e-93ad-aea61d9e038a)




## 3. 모델 설계

### 3.1 시행착오
|모델|방식|교훈|
|:---|:---|:---|
|action v0|mediapipe, LSTM|데이터 선정 및 전처리 중요성|
|object v0|labelme, YOLO-detect|모델 성능 개선 필요|
|object v0.1|YOLO-pose와 YOLO-detect|action 모델 필요|

## 3.2 최종 모델

### Action v1

#### 데이터 전처리

![Screenshot from 2024-06-17 16-27-41](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/86091697/d6b6da03-bea5-4113-b254-e2860c994140)


#### 모델 학습

![image](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/86091697/5e9c2440-c979-4c2f-a54e-8173adbc04d5)


#### 모델 평가

![image](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/86091697/2f4624c7-aad0-4667-85a4-dda690b95ce2)
![image](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/86091697/42a887f8-70a8-40e5-b8a2-0c009dbb4a15)


### Object v1

#### 데이터 전처리

![image](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/86091697/f5ab6f30-df98-4cf4-8439-ec8a77485fc2)

#### 모델 학습

![image-20240606-135248](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/86091697/4259be38-b05c-49a4-aa28-30a864999b63)
![image-20240606-135227](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/86091697/ee3fadad-7431-43c0-95e8-f10fd3bc0f6b)
#### 모델 평가

![image](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/86091697/fb546458-001f-4b26-be54-ea6f3bac16b9)
![image](https://github.com/addinedu-ros-5th/deeplearning-repo-1/assets/86091697/45cd42bf-a5e1-428e-b6f8-8bcdc5c804a5)

## 4. 프로젝트 시연


