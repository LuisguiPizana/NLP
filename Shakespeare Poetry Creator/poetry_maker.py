# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:32:44 2022

@author: Guillermo Pizana
"""


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Conv1D



#%%


def n_gram_seqs(corpus, tokenizer):
    
    input_sequences = []
    
    sentences = []
    
    for line in corpus:
        
        if len(line) != 0:
        
            sentences.append(line)
            
            vector = tokenizer.texts_to_sequences([line])
        
            input_sequences.append(vector[0])
    
    return input_sequences, sentences




def pad_seqs(input_sequences, maxlen):
    
    
    padded_seqs = pad_sequences(input_sequences, maxlen = maxlen)
    
    return padded_seqs



def features_and_labels(input_sequences, total_words):

    xs = input_sequences[:,:-1]
    
    labels = input_sequences[:,-1]

    labels = tf.keras.utils.to_categorical(labels, num_classes = total_words)

    return xs, labels




def create_model(total_words, max_sequence_len):
    
    model = Sequential()
    model.add(Embedding(total_words, 64))
    model.add(Conv1D(filters = 64, strides = 1, kernel_size = 5,activation = 'relu', input_shape = [max_sequence_len -1,1]))
    model.add(Bidirectional(LSTM(32, return_sequences = True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(total_words, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model


'''

#%%


# Define path for file with sonnets
sonet_file = 'sonets.txt'


# Read the data
with open(sonet_file) as f:
    corpus = f.read()

# Convert to lower case and save as a list
corpus = corpus.lower().split("\n")[:-1]

print(f"There are {len(corpus)} lines of sonnets\n")
print(f"The first 5 lines look like this:\n")
for i in range(5):
  print(corpus[i])


#%%



tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1


#%%

input_sequences, sentences = n_gram_seqs(corpus,tokenizer)


max_sequence_len = max([len(x) for x in input_sequences])



#%%

input_sequences = pad_seqs(input_sequences, max_sequence_len)

xs, ys = features_and_labels(input_sequences, total_words)


#%%

model = create_model(total_words, max_sequence_len)

print(model.summary)


#%%

history = model.fit(xs, ys, validation_split = .3 ,epochs = 40, verbose = 1, batch_size = 5)



#%%


acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation Loss')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()




#%%

#seed_text = "Thus my heart burned with a vivid love"
next_words = 100
  
for _ in range(next_words):
	# Convert the text into sequences
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	# Pad the sequences
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	# Get the probabilities of predicting a word
	predicted = model.predict(token_list, verbose=0)
	# Choose the next word based on the maximum probability
	predicted = np.argmax(predicted, axis=-1).item()
	# Get the actual word from the word index
	output_word = tokenizer.index_word[predicted]
	# Append to the current text
	seed_text += " " + output_word

print(seed_text)

'''


'''
Thus my heart burned with a vivid love
refusest fulfil this you live
me more, more forth lose rhyme dear
 weep change still is see thee mind days live
 mind eye brow rhyme dun room guard care brain
'''



'''

Thus my heart burned with a vivid love
 poverty use young prove sorrow lie enclose,
 set tend spent ruminate die rehearse
 argument worth bright see love say write
 write cross parts leave spring made spent
 green defeated abide argument bright hate
 seen tomb sing recite viewest forth pride
 commend unset sits possess'd room cheer
 own grow herd pride belong shore away now
 say write parts parts strong spring made 
 rage there difference change away seen say
 cross chest parts strong take same sport
 stay rehearse dead hate boast new hate so so life ' ' ' 
 alone alone more more lie strong mend set excellence day filed appear
 
 
    model = Sequential()
    model.add(Embedding(total_words, 64))
    model.add(Conv1D(filters = 64, strides = 1, kernel_size = 5,activation = 'relu', input_shape = [max_sequence_len -1,1]))
    model.add(Bidirectional(LSTM(32, return_sequences = True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(total_words, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    
    history = model.fit(xs, ys, validation_split = .3 ,epochs = 40, verbose = 1, batch_size = 5)


'''

