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

import numpy as np
import pandas as pd
import keras


# # time-series RNN

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.randn(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # 사인곡선1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # 사인곡선2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # 잡음
    return series[..., np.newaxis].astype(np.float32)


generate_time_series(10, 100).shape

n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, n_steps], series[9000:, -1]
series.shape

X_train

#naive prediction
y_pred = X_valid[:, -1]
np.mean(keras.losses.mean_sqared_error(y_valid, y_pred))

# 완전연결 네트워크
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape[50, 1]),
    keras.layers.Dense(1)
])




















































