# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: cook
#     language: python
#     name: cook
# ---

# +
# numpy로 RNN 구현하기

import numpy as np

timesteps = 100    # 입력 시퀀스에 있는 타임 스텝의 수
input_features = 32    # 입력 특성의 차원
output_features = 64   # 출력 특성의 차원

inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features, ))        # 초기상태 0

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs: # input_t.shape : (input_features,)
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0) # 최종 출력 크기는 (timesteps, output_features)인 2D 텐서.
# -

final_output_sequence.shape

np.dot(W, input_t)

# ## IMDB 문제 적용해보기

# +
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32

#데이터 로딩
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

print(f'''
~ 1. raw datasets ~
input_train.shape : {input_train.shape}
input_test.shape: {input_test.shape}
y_train : {y_train.shape}
y_test : {y_test.shape}

input_train sample : {input_train}

''')

print(len(input_train), '훈련 시퀀스')
print(len(input_test), '테스트 시퀀스')

print('시퀀스 패딩(sample x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)


print(f'''
~ 2. padding ~
input_train.shape : {input_train.shape}
input_test.shape: {input_test.shape}
y_train : {y_train.shape}
y_test : {y_test.shape}
''')

# +
# Embedding층과 SimpleRNN층을 사용해 간단한 순환 네트워크 훈련.
from keras.layers import Dense
from tensorflow.keras import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# +
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
print('epochs ', epochs)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs,val_acc, 'b', label='Validaton acc')
plt.title('Training, Validitaion accuracy')
plt.legend()
plt.show()
# -

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs,val_loss, 'b', label='Validaton loss')
plt.title('Training, Validitaion loss')
plt.legend()
plt.show()

y_pred = model.predict(input_test)
results = model.evaluate(input_test, y_test)
results # test loss / test accuracy


