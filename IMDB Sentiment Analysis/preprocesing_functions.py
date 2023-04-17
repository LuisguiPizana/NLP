import pandas as pd
import numpy as np
import nltk
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, Conv2D, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def clean_data(text):

    sentence = text.lower()

    sentence = sentence.replace('<br />', '')
    
    # Remove numbers, special characters and extra whitespaces
    sentence = re.sub(r"[^1-9A-Za-zñÑáéíóúÁÉÍÓÚüÜ']+", ' ', sentence)

    return sentence


def preprocess_data(text):

    text['review'] = text['review'].apply(clean_data)
    
    text['sentiment'] = text['sentiment'].apply(lambda x : 1 if x == 'positive' else 0)

    return text['review'].values, text['sentiment'].values

