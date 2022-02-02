from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
TfInput = Union[np.ndarray, tf.Tensor]
import dill

def build_feature_NN(num_units = 64):
    """ Generates a model to fit a specific feature of the dataset"""
    model = Sequential()

    # add input layer
    model.add(Dense(1))
    model.add(Dense(num_units, kernel_initializer='glorot_uniform', activation='relu'))
    # add output layer
    model.add(Dense(1))
    # compile computing MSE with Adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model

class NeuralGAM(tf.keras.Model):
    """
    Neural Generalized Additive Model.
    """

    def __init__(self,
                num_inputs,
                num_units,
                **kwargs):
        """Initializes NeuralGAM hyperparameters.

        Args:
            num_inputs: Number of feature inputs in input data.
            num_units: Number of hidden units in first layer of each feature net.
            **kwargs: Arbitrary keyword arguments. Used for passing the `activation`
            function as well as the `name_scope`.
        """
        super(NeuralGAM, self).__init__()
        self._num_inputs = num_inputs
        self._num_units = num_units
        self._kwargs = kwargs
        self.feature_networks = [None] * self._num_inputs

        self.build()
        
    def build(self):
        """Builds the FeatureCNNs"""
        self.feature_cnns = [None] * self._num_inputs
        for i in range(self._num_inputs):
            self.feature_cnns[i] = build_feature_NN(self._num_units)

        self._true = tf.constant(True, dtype=tf.bool)
        self._false = tf.constant(False, dtype=tf.bool)


    def fit(self, X_train, y_train, epochs, batch_size, max_iter):
        converged = False
        
        self.beta = y_train.mean()
            
        f = X_train*0
        index = f.columns.values
        CONVERGENCE_THRESHOLD = 0.0001
        self.training_mse = [] * max_iter
        it = 0
        
        # Make the data be zero-mean
        Z = y_train - self.beta
        
        # Track the squared error of the estimates
        while not converged and it < max_iter:
            #for each feature
            for k in range(len(X_train.columns)):
                idk = (list(range(0,k,1))+list(range(k+1,len(X_train.columns),1)))    # Get idx of columns != k
                
                # Compute the partial residual
                residuals = Z - f[index[idk]].sum(axis=1)
                # Fit network k with X_train[k] towards residuals
                self.feature_cnns[k].fit(X_train[X_train.columns[k]],residuals, epochs=epochs, batch_size=batch_size) 
                
                # Update f with current learned function for predictor k -- get f ready for compute residuals at next iteration
                f[index[k]] = self.feature_cnns[k].predict(X_train[X_train.columns[k]])  
            
            #compute how far we are from estimating y_train
            err = mean_squared_error(Z, f.sum(axis=1))
            self.training_mse.append(err)
            mse_delta = np.abs(self.training_mse[it] - self.training_mse[it-1])
            print("ITERATION#{0}: Current MSE = {1}".format(it, self.training_mse[it]))
            print("ITERATION#{0}: MSE delta with prev iteration = {1}".format(it, mse_delta))
            
            if err < CONVERGENCE_THRESHOLD:
                converged = True
            
            it+=1
            
        # Reconstruct y = sum(f) + beta
        self.y = self.beta + f.sum(axis=1)
        
        return self.y, self.training_mse

        
    def predict(self, X: pd.DataFrame):
        """Computes Neural GAM output by computing a linear combination of the outputs of individual feature networks."""
        
        output = pd.DataFrame(columns=range(len(X.columns)))
        for i in range(len(X.columns)):
            output[i] = pd.Series(self.feature_cnns[i].predict(X[i]).flatten())
            
        y_pred = output.sum(axis=1)  + self.beta
        return y_pred
    
    def save_model(self, output_path):
        with open(output_path, "wb") as file:
            dill.dump(self, file, dill.HIGHEST_PROTOCOL)
            
def load_model(model_path):
    with open(model_path, "rb") as file:
        return dill.load(file)