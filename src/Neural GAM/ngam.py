import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, LayerNormalization, Activation
import tensorflow as tf
import pandas as pd


def build_feature_NN(num_units = 64):
    """ Generates a model to fit a specific feature of the dataset"""
    model = Sequential()

    # add input layer
    model.add(Dense(1, activation='linear', use_bias=False,))
    model.add(Dense(num_units, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(num_units/2, kernel_initializer='glorot_uniform', activation='relu'))
    
    # add output layer
    model.add(Dense(1))

    # compile computing MSE with Adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    
    return model


class Neural_GAM(tf.keras.Model):
    """
    Neural additive model.
        num_inputs: Number of feature inputs in input data.
        num_units: Number of hidden units in first layer of each feature network.
    """
    def __init__(self,
               num_inputs,
               num_units,
               beta,
               **kwargs):
   
        super(Neural_GAM, self).__init__()
        self._num_inputs = num_inputs
        self._num_units = num_units
        self._kwargs = kwargs
        self._beta = beta
    
    def build(self):
        self.feature_networks = [None] * self._num_inputs
        for i in range(self._num_inputs):  
            self.feature_networks[i] = build_feature_NN(self._num_units)

        self._bias = self.add_weight(
            name='bias',
            initializer=self._beta,
            shape=(1,),
            trainable=False)
    
    def train(self, X_train, y_train, epochs, batch_size):
        converged = False
        alpha = y_train.mean()
            
        f = X_train*0
        index = f.columns.values
        g = f.copy()
        CONVERGENCE_THRESHOLD = 0.01
        mse_delta = (y_train * y_train).mean()
        prev_mse = 0
        it = 0
        
        Z = y_train - self.beta
        
        # Track the squared error of the estimates
        while not converged:
            #for each column
            for k in range(len(X_train.columns)):
                idk = (list(range(0,k,1))+list(range(k+1,len(X_train.columns),1)))    # Get idx of columns != k
                ep = y_train - f[index[idk]].sum(axis=1) - alpha #set ep for current function (substract contribution of other features from y)

                self.feature_networks[k].fit(X_train[k],ep, epochs=epochs, batch_size=batch_size) 
                newy = self.feature_networks[k].predict(X_train[k])  #Get current learned y for predictor k

                g[index[k]] = newy
                f = g
            
            
            residuals = y_train - f.sum(axis=1)
            
            Z0 = Z
            Z = Z - residuals
            
            mse = mean_squared_error(Z, Z0)
            mse_delta = np.abs(Z - Z0)
            print("ITERATION#{0}: Current MSE = {1}".format(it, mse))
            print("ITERATION#{0}: MSE delta with prev iteration = {1}".format(it, mse_delta))
            
            if Z < CONVERGENCE_THRESHOLD:
                converged = True
            
            it+=1
            
        # Reconstruct y
        ycal = alpha + f.sum(axis=1)
        
        return ycal, alpha

        
    def compute(self, X: pd.DataFrame):
        """Computes Neural GAM output by computing a linear combination of the outputs of individual feature networks."""
        
        output = pd.DataFrame(columns=range(len(X.columns)))
        for i in range(len(X.columns)):
            output[i] = pd.Series(self.feature_networks[i].predict(X[i]).flatten()) + self._beta
            
        y_pred = output.sum(axis=1)
        return y_pred


    
    
    