import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [[series[i+r] for r in range(window_size)] for i in range(len(series)-window_size)]
    y = [series[i] for i in range(window_size, len(series))]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:window_size])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))
    
    return model



### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    return ''.join([c for c in text if c in punctuation or c.islower() or c.isspace()])

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [[text[i+c] for c in range(window_size)] for i in range(0, len(text)-window_size, step_size)]

    outputs = [text[i] for i in range(window_size, len(text), step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    '''
    layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
layer 3 should be a softmax activation ( since we are solving a multiclass classification)
Use the categorical_crossentropy loss
    '''
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation('softmax'))
    return model
    
    