import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define neural network 


def build_neural_network():
    model = Sequential()

    model.add(Dense(40, input_dim=1, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    
def fit(model, X, y, epochs=50, batch_size=50)
    model.fit(X, y, epochs, batch_size)

def predict(model, X, y):
    

