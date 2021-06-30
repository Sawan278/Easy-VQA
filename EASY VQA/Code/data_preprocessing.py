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

def setup(use_data_dir):
    print('\n--- Reading questions...')
    # Read data from data/ folder
    def read_questions(path):
        with open(path, 'r') as file:
            qs = json.load(file)
        texts = [q[0] for q in qs]
        answers = [q[1] for q in qs]
        image_ids = [q[2] for q in qs]
        return (texts, answers, image_ids)
    
    train_qs, train_answers, train_image_ids = read_questions('data/train/questions.json')
    valid_qs, valid_answers, valid_image_ids = read_questions('data/validation/questions.json')
    test_qs, test_answers, test_image_ids = read_questions('data/test/questions.json')
  
    print(f'Read {len(train_qs)} training questions, {len(valid_qs)} validation questions and {len(test_qs)} testing questions.')

    print('\n--- Reading answers...')
    # Read answers from data/ folder
    with open('data/answers.txt', 'r') as file:
        all_answers = [a.strip() for a in file]
    
    num_answers = len(all_answers)
    print(f'Found {num_answers} total answers:')
    print(all_answers)

    print('\n--- Reading/processing images...')
    def load_and_proccess_image(image_path):
        # Load image, then scale and shift pixel values to [-0.5, 0.5]
        im = img_to_array(load_img(image_path))
        return im / 255 - 0.5

    def read_images(paths):
        # paths is a dict mapping image ID to image path
        # Returns a dict mapping image ID to the processed image
        ims = {}
        for image_id, image_path in paths.items():
            ims[image_id] = load_and_proccess_image(image_path)
        return ims

    # Read images from data/ folder
    def extract_paths(dir):
        paths = {}
        for filename in os.listdir(dir):
            if filename.endswith('.png'):
                image_id = int(filename[:-4])
                paths[image_id] = os.path.join(dir, filename)
        return paths

    train_ims = read_images(extract_paths('data/train/images'))
    valid_ims = read_images(extract_paths('data/validation/images'))
    test_ims  = read_images(extract_paths('data/test/images'))
  
    im_shape = train_ims[0].shape
    print(f'Read {len(train_ims)} training images, {len(valid_ims)} validation images and {len(test_ims)} testing images.')
    print(f'Each image has shape {im_shape}.')


    print('\n--- Fitting question tokenizer...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_qs)

    # We add one because the Keras Tokenizer reserves index 0 and never uses it.
    vocab_size = len(tokenizer.word_index) + 1
    print(f'Vocab Size: {vocab_size}')
    print(tokenizer.word_index)


    print('\n--- Converting questions to bags of words...')
    train_X_seqs = tokenizer.texts_to_matrix(train_qs)
    valid_X_seqs = tokenizer.texts_to_matrix(valid_qs)
    test_X_seqs = tokenizer.texts_to_matrix(test_qs)
    print(f'Example question bag of words: {train_X_seqs[0]}')


    print('\n--- Creating model input images...')
    train_X_ims = np.array([train_ims[id] for id in train_image_ids])
    valid_X_ims = np.array([valid_ims[id] for id in valid_image_ids])
    test_X_ims = np.array([test_ims[id] for id in test_image_ids])


    print('\n--- Creating model outputs...')
    train_answer_indices = [all_answers.index(a) for a in train_answers]
    valid_answer_indices = [all_answers.index(a) for a in valid_answers]
    test_answer_indices = [all_answers.index(a) for a in test_answers]
  
    train_Y = to_categorical(train_answer_indices)
    valid_Y = to_categorical(valid_answer_indices)
    test_Y = to_categorical(test_answer_indices)
    print(f'Example model output: {train_Y[0]}')

    return (train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs,
          test_Y, im_shape, vocab_size, num_answers,
          all_answers, test_qs, test_answer_indices, valid_X_ims, valid_X_seqs, valid_Y)