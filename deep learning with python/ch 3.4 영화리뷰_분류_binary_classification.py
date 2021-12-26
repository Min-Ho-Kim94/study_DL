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

# train compile 단계에서 고려할 3가지
#
#     1. loss function
#     2. optimizer
#     3. metrics

# # ch. 3.4 IMDB dataset

# +
import numpy as np
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb\
.load_data(num_words=10000)
# -

print(f'''
train_data.shape : {train_data.shape}
test_data.shape : {test_data.shape}

train_labels.shape : {train_labels.shape}
test_labels.shape: : {test_labels.shape}
''')

train_data[[0]]

# 0 : 긍정 / 1 : 부정
train_labels

# 원래 영어단어 사전 불러와서 mapping(decoding)
word_index = imdb.get_word_index()
reverse_word_index = dict([
    (value, key) for (key, value) in word_index.items()
])
decoded_review = ' '.join([
    reverse_word_index.get(i - 3, '?') for i in train_data[0]
])


decoded_review


# ## 1 데이터 준비 (3.4.2)
#
# - list to tensor (1D -> 2D)
#
#     여기서는 one hot encoding 이용

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
x_train[0]

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
y_train

print(f'''
x_train.shape : {x_train.shape}
x_test.shape : {x_test.shape}

y_train.shape : {y_train.shape}
y_test.shape: : {y_test.shape}
''')

# ## 2 모델 정의.
#
# 1. Affine - Relu (hidden size = 16)
# 2. Affine - Relu (hidden size = 16)
# 3. Sigmoid

# +
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# -

# ## 3 모델 컴파일
#
# 1. optimizer : rmsprop
# 2. loss : binary_crossentropy
# 3. metrics : accuracy

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# +
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
# -

# ## 4 훈련 검증

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

x_val.shape, partial_x_train.shape

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4, # 20
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

# +
import matplotlib.pyplot as plt

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# +
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()
# -

results = model.evaluate(x_test, y_test)

results # test loss / test accuracy




