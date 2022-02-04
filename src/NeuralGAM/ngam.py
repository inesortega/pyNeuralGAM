from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
TfInput = Union[np.ndarray, tf.Tensor]
import dill

def build_feature_NN(num_units = 64):
    """ Generates a model to fit a specific feature of the dataset"""
    model = Sequential()

    # add input layer
    model.add(Dense(1))
    # The Hidden Layers :
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
        self.training_mse = list()

        self.build()
        
    def build(self):
        """Builds a FeatureNNs for each feature """
        for i in range(self._num_inputs):
            self.feature_networks[i] = build_feature_NN(self._num_units)

    def fit(self, X_train, y_train, max_iter):
        converged = False
        
        self.beta = y_train.mean()
            
        f = X_train*0
        index = f.columns.values
        CONVERGENCE_THRESHOLD = 0.001
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
                self.feature_networks[k].fit(X_train[X_train.columns[k]],residuals, epochs=1) 
                
                # Update f with current learned function for predictor k -- get f ready for compute residuals at next iteration
                f[index[k]] = self.feature_networks[k].predict(X_train[X_train.columns[k]])
                f[index[k]] = f[index[k]] - np.mean(f[index[k]])  
            
            #compute how far we are from estimating y_train
            err = mean_squared_error(Z, f.sum(axis=1))
            self.training_mse.append(err)
            mse_delta = np.abs(self.training_mse[it] - self.training_mse[it-1])
            print("ITERATION#{0}: Current MSE = {1}".format(it, self.training_mse[it]))
            print("ITERATION#{0}: MSE delta with prev iteration = {1}".format(it, mse_delta))
            
            if mse_delta < CONVERGENCE_THRESHOLD and it > 0:
                converged = True
            
            it+=1
            
        # Reconstruct y = sum(f) + beta
        y = self.beta + f.sum(axis=1)
        
        return y, self.training_mse

    def get_partial_dependencies(self, X: pd.DataFrame):
        """ Compute the partial dependencies for each feature in X"""
        output = pd.DataFrame(columns=range(len(X.columns)))
        for i in range(len(X.columns)):
            output[i] = pd.Series(self.feature_networks[i].predict(X[X.columns[i]]).flatten())
        return output
            
    def predict(self, X: pd.DataFrame):
        """Computes Neural GAM output by computing a linear combination of the outputs of individual feature networks."""
        output = self.get_partial_dependencies(X)
        y_pred = output.sum(axis=1)  + self.beta
        return y_pred
    
    def save_model(self, output_path):
        with open(output_path, "wb") as file:
            dill.dump(self, file, dill.HIGHEST_PROTOCOL)
       
          
def load_model(model_path) -> NeuralGAM:
    with open(model_path, "rb") as file:
        return dill.load(file)