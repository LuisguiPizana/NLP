# Poetry Maker - Shakespeare Sonnets
This project creates a poetry generator using a deep learning model, trained on a dataset of Shakespeare's sonnets. It consists of three scripts:

- Main script for training and testing the model (poetry_maker.py)
- Script for obtaining the data (sonnets)
- Script for optimizing the hyperparameters of the model

## Main Script
The main script consists of several steps:

Importing required libraries and defining utility functions for data preprocessing and model creation
Reading the dataset from the 'sonets.txt' file and preprocessing the text
Tokenizing the text and generating input sequences
Padding the sequences and creating features and labels
Building the deep learning model using TensorFlow and Keras
Training the model and plotting training accuracy and loss
Generating new poetry by providing a seed text

## Data Obtaining Script
The script for obtaining the data does the following:

Imports required libraries and defines functions for extracting and cleaning sonnets
Uses Selenium and BeautifulSoup to crawl the sonnet data from the Open Source Shakespeare website
Writes the cleaned sonnets to the 'sonets.txt' file

## Hyperparameter Optimization Script
The optimization script utilizes the Optuna library to find the best hyperparameters for the model. It performs the following steps:

Imports required libraries and loads the preprocessed data
Defines the objective function that trains the model with different hyperparameters and returns the validation accuracy
Creates an Optuna study with a direction to maximize the objective function
Optimizes the objective function with a specified number of trials
Prints the best trial and the best set of hyperparameters

## Usage
Run the data obtaining script to scrape the sonnets and save them to the 'sonets.txt' file.
Run the hyperparameter optimization script to find the best hyperparameters for the model.
Use the main script to train the model with the optimized hyperparameters and test it by generating new poetry.

## Requirements
Python 3.x
TensorFlow 2.x
Keras
NumPy
Matplotlib
Selenium
BeautifulSoup
Optuna
Pandas
Webdriver Manager