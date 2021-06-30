from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
def build_model(im_shape, vocab_size, num_answers):
    #The CNN model
    im_input = Input(shape=im_shape)
    x1 = Conv2D(8, 3, padding='same')(im_input)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(16, 3, padding='same')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Flatten()(x1)
    # Adding two fully-connected layers after the CNN for good measure
    x1 = Dense(32, activation='tanh')(x1)
    x1 = Dense(32, activation='tanh')(x1)

    # A simple question network formed by three fully-connected layers
    q_input = Input(shape=(vocab_size,))
    x2 = Dense(32, activation='tanh')(q_input)
    x2 = Dense(32, activation='tanh')(x2)
    x2 = Dense(32, activation='tanh')(x2)

    # The processed image and the processed question is merged
    out = Multiply()([x1, x2])
    out = Dense(32, activation='tanh')(out)
    out = Dense(32, activation='tanh')(out)
    out = Dense(num_answers, activation='softmax')(out)

    model = Model(inputs=[im_input, q_input], outputs=out)
    model.compile(Adam(learning_rate=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model