# English-Spanish Neural Machine Translation with Attention
This project focuses on training a neural machine translation (NMT) model to translate English text to Spanish using a sequence-to-sequence model with attention mechanism. The model is implemented using TensorFlow and Keras, and the training data is read from a large text file using a custom data generator.

## Table of Contents
1. [Problem Definition](#problem-definition)
2. [Model Architecture](#model-architecture)
3. [Data Ingestion](#data-ingestion)
4. [Training](#training)
5. [Resources](#resources)

## Problem Definition <a name="problem-definition"></a>
Neural machine translation is an application of deep learning to automatically translate text from one language to another1. The primary goal is to build a model capable of understanding the input text in the source language and generating an accurate translation in the target language.

This project utilizes a sequence-to-sequence (Seq2Seq) model architecture, which is widely used for NMT tasks[^2^]. Seq2Seq models consist of two main components: an encoder and a decoder. The encoder reads the input text and generates a fixed-size context vector, while the decoder uses this context vector to generate the translated text.

The attention mechanism is employed in this project to address the limitation of fixed-size context vectors in capturing long-range dependencies[^3^]. Attention allows the model to focus on different parts of the input sequence while generating each word in the output sequence, improving the translation quality[^3^].

## Model Architecture <a name="model-architecture"></a>
The Seq2SeqAttention model is implemented using the following layers:

### Encoder:

Input layer: takes in English word sequences
Embedding layer: maps words to embeddings
LSTM layer: processes the input sequence and generates hidden states
Attention Mechanism:

Implements the Bahdanau attention mechanism to compute context vectors using the encoder's hidden states and the decoder's previous hidden state[^4^]

### Decoder:

Input layer: takes in Spanish word sequences (shifted by one position)
Embedding layer: maps words to embeddings
LSTM layer: processes the input sequence, encoder's final hidden states, and the context vector from the attention mechanism
Dense output layer: produces the probability distribution over the Spanish vocabulary
Data Ingestion
The training data is read from a large text file using a custom data generator called MemoryDataGenerator. This generator implements the following features:

Reservoir Sampling: A randomized algorithm for selecting a simple random sample of k items from a list containing n items, where n is either very large or unknown[^5^]. This method is memory-efficient and ensures each item in the input list has an equal probability of being included in the final sample[^5^].

Tokenizer Fitting: English and Spanish tokenizers are fitted on a random subset of the data using reservoir sampling. This step reduces memory usage and speeds up the training process.

Batch Generation: The generator reads the training data line by line and processes it using a ThreadPoolExecutor for parallel processing. The input sequences are tokenized, padded, and batched before being fed into the model.

## Training <a name="training"></a>
The model is trained using the custom data generator, which reads and processes the data in parallel. This approach allows the model to handle large datasets that may not fit into memory.

After each epoch, the data generator shuffles the training data to ensure the model is exposed to different samples in each epoch.

## Resources <a name="resources"></a>

Sequence to Sequence Learning with Neural Networks by Sutskever, I., Vinyals, O., & Le, Q. V. (2014): [^1^]
https://papers.nips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf

Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation by Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014): [^2^]
https://www.aclweb.org/anthology/D14-1179/

Neural Machine Translation by Jointly Learning to Align and Translate by Bahdanau, D., Cho, K., & Bengio, Y. (2015): [^3^]
https://arxiv.org/abs/1409.0473

Effective Approaches to Attention-based Neural Machine Translation by Luong, M. T., Pham, H., & Manning, C. D. (2015): [^4^]
https://www.aclweb.org/anthology/D15-1166/

Random Sampling with a Reservoir by Vitter, J. S. (1985): [^5^]
https://doi.org/10.1145/3147.3165

