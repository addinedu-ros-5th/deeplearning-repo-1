import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# CSV 파일 읽기
df = pd.read_csv('deeplearning_demo.csv')

# 시퀀스 길이 설정 (예: 36)
sequence_length = 36

# 시퀀스를 생성하는 함수
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length + 1):
        seq = data.iloc[i:i + seq_length, :-1].values  # 시퀀스 데이터
        label = data.iloc[i + seq_length - 1, -1]      # 시퀀스의 마지막 라벨
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# 시퀀스 생성
X, y = create_sequences(df, sequence_length)

# 레이블 인코딩 (1 -> 0, 4 -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 고유한 클래스 수 확인
num_classes = len(np.unique(y_encoded))

# 원-핫 인코딩
y_one_hot = to_categorical(y_encoded, num_classes=num_classes)

# 학습 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(100, input_shape=(sequence_length, 34), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))  # 다중 클래스 분류를 위한 소프트맥스 활성화 함수 사용

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 모델 학습
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# 예측 수행
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# 예측 결과 출력
print(f'Predicted classes: {y_pred_classes}')
print(f'True classes: {y_test}')

# 모델 저장
model.save('lstm_model_demo.keras')
print("모델이 'lstm_model_yohan.keras' 파일로 저장되었습니다.")