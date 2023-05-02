# IMDB Sentiment Analysis with LSTM and Hyperparameter Optimization
This project focuses on training a sentiment analysis model to classify movie reviews from the IMDB dataset as positive or negative using a deep learning approach. The model is implemented using TensorFlow and Keras, and the training data is preprocessed using a custom cleaning function. Hyperparameter optimization is performed using Keras Tuner and the Hyperband algorithm.

This project is a work in progress. The final results of the optimization are not finished yet. 

## Table of Contents
Problem Definition
Model Architecture
Data Preprocessing
Hyperparameter Optimization
Training
Results
Resources

Problem Definition <a name="problem-definition"></a>
Sentiment analysis is a natural language processing task that aims to determine the sentiment or emotion expressed in a piece of text<sup>[1]</sup>. In this project, the goal is to build a model capable of understanding movie reviews and predicting whether the sentiment expressed in the review is positive or negative.

Model Architecture <a name="model-architecture"></a>
The sentiment analysis model is implemented using the following layers as a basis:

Input layer: takes in the preprocessed movie review sequences
Embedding layer: maps words to embeddings
LSTM layer: processes the input sequence and generates hidden states
Optional Bidirectional, Conv1D, and Dropout layers: to explore different architecture variations
Dense layer: fully connected layer with ReLU activation
Output layer: produces the probability of the review being positive using sigmoid activation

It is important to note that the architecure might change with the optimization.


Data Preprocessing <a name="data-preprocessing"></a>
The IMDB movie reviews dataset is preprocessed using the following steps:

Cleaning: removing special characters, numbers, and extra whitespaces; converting text to lowercase
Tokenization: splitting text into words and mapping them to integer tokens
Padding: ensuring all input sequences have the same length by padding with zeros
Hyperparameter Optimization <a name="hyperparameter-optimization"></a>
The hyperparameters of the model are optimized using Keras Tuner, the Hyperband and bayesian optimization algorithms. The hyperparameters include:

Number of LSTM units
Size of the embedding layer
Optional addition of Bidirectional, Conv1D, and Dropout layers
The Hyperband algorithm is used to efficiently explore the hyperparameter space and find the best combination of hyperparameters.

Training <a name="training"></a>
The model is trained using the preprocessed data and the best combination of hyperparameters found during the optimization process. Training is performed using the Adam optimizer, binary cross-entropy loss, and accuracy metric.

Results <a name="results"></a>
Pending.

Resources <a name="resources"></a>
Learning Word Vectors for Sentiment Analysis by Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011): <sup>[1]</sup>
https://www.aclweb.org/anthology/P11-1015/

Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization by Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018):
https://arxiv.org/abs/1603.06560

Keras Tuner: Hyperparameter Tuning for Humans by Ouali, A., Passricha, R., & Ravanelli, M. (2020):
https://arxiv.org/abs/2009.08454