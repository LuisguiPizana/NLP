import time

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

#Counts number of repetitions of elements in a list
from collections import Counter


from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile

dataset_folder_path = 'data'
dataset_filename = 'text8.zip'
dataset_name = 'Text8 Dataset'


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def download_and_read_text():

    if not isfile(dataset_filename):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc=dataset_name) as pbar:
            urlretrieve(
                'http://mattmahoney.net/dc/text8.zip',
                dataset_filename,
                pbar.hook)

    if not isdir(dataset_folder_path):
        with zipfile.ZipFile(dataset_filename) as zip_ref:
            zip_ref.extractall(dataset_folder_path)
            
    with open('data/text8') as f:
        text = f.read()


    return text



def clean_text(text_string, min_num_appearances = 5):
    # Replace punctuation with tokens so we can use them in our model
    text = text_string.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in tqdm(words) if word_counts[word] > min_num_appearances]

    return trimmed_words




def fit_tokenizer(cleaned_text_list):
    #This function is just to eliminate the training of the Tokenizer object. Big drawbacks
    #we cannot change hyperparams.

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(cleaned_text_list)

    return tokenizer






def single_word_pair(text_list, index, window_size = 5):
    
    num_words = len(text_list)

    rand_window = np.random.randint(1, window_size +1)

    start_index = (index - rand_window) if (index - rand_window) > 0 else 0 

    end_index = (index + rand_window) if (index + rand_window) < num_words else num_words - 1

    word_list = set(text_list[start_index : index] + text_list[index + 1 : end_index])
    
    pairs = [(text_list[index], word) for word in word_list]

    return pairs



def word_pair(text_list, window_size = 5):

    word_pairs = []

    for word_index in tqdm(range(len(text_list))):

        word_pairs += single_word_pair(text_list, word_index, window_size = window_size)

    word_pairs = np.array(word_pairs)

    return word_pairs