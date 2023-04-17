# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:26:12 2022

@author: Guillermo Pizana
"""

import optuna

import poetry_maker as pm




import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Conv1D



#%%

def objective(trial):
    
    
    sonet_file = 'sonets.txt'


    # Read the data
    with open(sonet_file) as f:
        corpus = f.read()

    # Convert to lower case and save as a list
    corpus = corpus.lower().split("\n")[:-1]
    
    
    
    
    sonet_file = 'sonets.txt'


    # Read the data
    with open(sonet_file) as f:
        corpus = f.read()

    # Convert to lower case and save as a list
    corpus = corpus.lower().split("\n")[:-1]
    
    
    
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
        
        
    input_sequences, sentences = pm.n_gram_seqs(corpus,tokenizer)


    max_sequence_len = max([len(x) for x in input_sequences])   
    
    
    input_sequences = pm.pad_seqs(input_sequences, max_sequence_len)

    xs, ys = pm.features_and_labels(input_sequences, total_words)
    
    
    
    
    epochs = trial.suggest_int('epochs', low = 20, high = 150, step = 10)
    
    embedding_dims = trial.suggest_int('embedding_dims', low = 20, high = 150, step = 10)
    
    conv1_filters = trial.suggest_int('n_filters' , low = 20, high = 80, step = 10)
    
    kernel_size = trial.suggest_int('kernel_size', low = 2, high = 8, step = 2)
    
    lstm1 = trial.suggest_int('n_units_lstm1', low = 16, high = 126, step = 10)
    
    lstm2 = trial.suggest_int('n_units_lstm2', low = 16, high = 126, step = 10)
    
    batch_size = trial.suggest_int('batch_size', low = 3, high = 25, step = 2)
    

    model = Sequential()
    model.add(Embedding(total_words, embedding_dims))
    model.add(Conv1D(filters = conv1_filters, strides = 1, kernel_size = kernel_size,activation = 'relu', input_shape = [max_sequence_len -1,1]))
    model.add(Bidirectional(LSTM(lstm1, return_sequences = True)))
    model.add(Bidirectional(LSTM(lstm2)))
    model.add(Dense(total_words, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    
    history = model.fit(xs, ys, validation_split = .3 ,epochs = epochs, verbose = 1, batch_size = batch_size)


    val_acc = history.history['val_accuracy'][-1]

    # Number of components for dimentionality reduction
    
    
    
    '''
    FrozenTrial(number=8, values=[0.01697530783712864], datetime_start=datetime.datetime(2022, 9, 4, 15, 41, 48, 836534), datetime_complete=datetime.datetime(2022, 9, 4, 15, 44, 1, 879282),
                params={'epochs': 40,
                        'embedding_dims': 120,
                        'n_filters': 80,
                        'kernel_size': 6,
                        'n_units_lstm1': 126,
                        'n_units_lstm2': 16,
                        'batch_size': 25},
                distributions={'epochs': IntUniformDistribution(high=150, low=20, step=10),
                               'embedding_dims': IntUniformDistribution(high=150, low=20, step=10),
                               'n_filters': IntUniformDistribution(high=80, low=20, step=10),
                               'kernel_size': IntUniformDistribution(high=8, low=2, step=2),
                               'n_units_lstm1': IntUniformDistribution(high=126, low=16, step=10),
                               'n_units_lstm2': IntUniformDistribution(high=126, low=16, step=10),
                               'batch_size': IntUniformDistribution(high=25, low=3, step=2)},
                user_attrs={}, system_attrs={}, intermediate_values={}, trial_id=8, state=TrialState.COMPLETE, value=None)
    {'epochs': 40, 'embedding_dims': 120, 'n_filters': 80, 'kernel_size': 6, 'n_units_lstm1': 126, 'n_units_lstm2': 16, 'batch_size': 25}
    
    '''
    
    
    
    
    return val_acc



#%%


study = optuna.create_study(direction = 'maximize')

study.optimize(objective, n_trials = 10)


print(study.best_trial)


print(study.best_params)



#optuna.visualization.plot_optimization_history(study)

#optuna.visualization.plot_slice(study)

