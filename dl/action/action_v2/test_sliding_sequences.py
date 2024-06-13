from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 저장된 LSTM 모델 파일 경로
model_path = 'lstm_model.keras'

# 모델 불러오기
lstm_model = load_model(model_path)

# CSV 파일 읽기
df = pd.read_csv('deeplearning_demo.csv')

# 시퀀스 길이 설정 (예: 10)
sequence_length = 10

# 시퀀스를 생성하는 함수 (슬라이딩 윈도우 방식)
def create_sliding_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length + 1):
        seq = data.iloc[i:i + seq_length, :-1].values  # 시퀀스 데이터
        label = data.iloc[i + seq_length - 1, -1]      # 시퀀스의 마지막 라벨
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# 시퀀스 생성 (슬라이딩 윈도우 방식)
X, y = create_sliding_sequences(df, sequence_length)

# 시퀀스 길이에 맞추어 데이터 정제 (예: 패딩 추가)
X = pad_sequences(X, maxlen=sequence_length, dtype='float32', padding='post')

# 레이블 인코딩 (1 -> 0, 4 -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 고유한 클래스 수 확인
num_classes = len(np.unique(y_encoded))

# 원-핫 인코딩
y_one_hot = to_categorical(y_encoded, num_classes=num_classes)

# LSTM 모델 입력 형식 맞추기
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
X_train = X_train.reshape((-1, sequence_length, 34))  # 34는 스켈레톤 좌표의 특성 수입니다.
X_test = X_test.reshape((-1, sequence_length, 34))

# 모델 평가
loss, accuracy = lstm_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# 예측 수행
y_pred = lstm_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 예측 결과 출력
print(f'Predicted classes: {y_pred_classes}')
print(f'True classes: {np.argmax(y_test, axis=1)}')
