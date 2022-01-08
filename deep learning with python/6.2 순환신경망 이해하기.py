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
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0)
# -

final_output_sequence.shape


