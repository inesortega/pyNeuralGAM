from copy import deepcopy
from operator import inv
from typing import Union
from matplotlib.axis import YAxis
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1_l2
TfInput = Union[np.ndarray, tf.Tensor]
import dill
import os
from sklearn.preprocessing import MinMaxScaler
from mlflow.pyfunc import PythonModel 

class NeuralGAM(tf.keras.Model):
    """
    Neural Generalized Additive Model.
    """

    def __init__(self,
                num_inputs,
                link,
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
        self.link = link
        self._kwargs = kwargs
        self.feature_networks = [None] * self._num_inputs
        self.training_mse = list()
        self.y = None
        self.build()

    
    def build_feature_NN(self):
        """ Generates a model to fit a specific feature of the dataset"""
        model = Sequential()

        # add input layer
        model.add(Dense(1))
        # The Hidden Layers :       
        model.add(Dense(512, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(256, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(128, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(64, kernel_initializer='glorot_normal', activation='relu'))
        # add output layer
        model.add(Dense(1))
        # compile computing MSE with Adam optimizer
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mean_absolute_error'])
        
        return model


    def build(self):
        """Builds a FeatureNNs for each feature """
        for i in range(self._num_inputs):
            self.feature_networks[i] = self.build_feature_NN()

    def fit(self, X_train, y_train, max_iter, convergence_threshold):
        
        #Initialization
        converged = False
        f = X_train*0
        g = f
        index = f.columns.values
        DELTA_THRESHOLD = 0.001
        it = 0
        
        self.muhat = y_train.mean()
        self.eta = inv_link(self.link, self.muhat)
        Z = inv_link(self.link, y_train) - self.eta
        
        # Start backfitting algorithm
        while not converged and it < max_iter:
            #for each feature
            for k in range(len(X_train.columns)):
                idk = (list(range(0,k,1))+list(range(k+1,len(X_train.columns),1)))    # Get idx of columns != k
                
                # Compute the partial residual - remove from y the contribution from other features
                residuals = Z - g[index[idk]].sum(axis=1)
                
                # Fit network k with X_train[k] towards residuals
                self.feature_networks[k].fit(X_train[X_train.columns[k]], residuals, epochs=1) 
                
                # Update f with current learned function for predictor k -- get f ready for compute residuals at next iteration
                f[index[k]] = self.feature_networks[k].predict(X_train[X_train.columns[k]])
                f[index[k]] = f[index[k]] - np.mean(f[index[k]])  
            
            # update current estimations
            g = f
            
            #compute how far we are from estimating y_train
            y_ = apply_link(self.link, g.sum(axis=1) + self.eta)
            err = mean_squared_error(y_train, y_)          
            self.training_mse.append(err)
            mse_delta = np.abs(self.training_mse[it] - self.training_mse[it-1])          
            print("ITERATION#{0}: Current MSE = {1}".format(it, err))
            print("ITERATION#{0}: MSE delta with prev iteration = {1}".format(it, mse_delta))
            
            if (err < convergence_threshold or mse_delta < DELTA_THRESHOLD) and it > 0:
                print("Z and f(x) converged...")
                converged = True
                print("Achieved RMSE during training = {0}".format(mean_squared_error(y_train, y_, squared=False)))
                
            it+=1
        
        # Reconstruct learnt y
        self.eta = apply_link(self.link, self.eta)
        self.y = apply_link(self.link, g.sum(axis=1)) + self.eta
            
        return

    def get_partial_dependencies(self, X: pd.DataFrame, xform=True):
        """ 
        Get the partial dependencies for each feature in X
        xform : bool, default: True, whether to apply the inverse link function and return values
                on the scale of the distribution mean (True), or to keep on the linear predictor scale (False)
        """
        output = pd.DataFrame(columns=range(len(X.columns)))
        for i in range(len(X.columns)):
            output[i] = pd.Series(self.feature_networks[i].predict(X[X.columns[i]]).flatten())
        if xform:
            output = output.apply(lambda x: apply_link(self.link, x))
        return output
            
    def predict(self, X: pd.DataFrame):
        """Computes Neural GAM output by computing a linear combination of the outputs of individual feature networks."""
        output = self.get_partial_dependencies(X, xform=False)
        y = apply_link(self.link, output.sum(axis=1) + self.eta)
        return y
        
    
    def save_model(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if os.path.exists(output_path + "/model.ngam"):
            os.remove(output_path + "/model.ngam")
        #with open(output_path + "/model.ngam", "wb") as file:
            #dill.dump(self, file, dill.HIGHEST_PROTOCOL)
        import mlflow
        mlflow.pyfunc.save_model(python_model=self, path=output_path + "/model.ngam")
        return output_path + "/model.ngam"
      
    
def compute_loss(type, actual, pred):
    # calculate binary cross entropy
    if type == "binary_cross_entropy":
        sum_score = 0.0
        for i in range(len(actual)):
            sum_score += actual[i] * np.log(1e-15 + pred[i])
        mean_sum_score = 1.0 / len(actual) * sum_score
        return -mean_sum_score
    elif type == "mse":
        return mean_squared_error(actual, pred)    
    elif type == "rmse":
        return mean_squared_error(actual, pred, squared=True)
    elif type == "categorical_cross_entropy":
        sum_score = 0.0
        for i in range(len(actual)):
            for j in range(len(actual[i])):
                sum_score += actual[i][j] * np.log(1e-15 + pred[i][j])
        mean_sum_score = 1.0 / len(actual) * sum_score
        return -mean_sum_score
    else:
        raise ValueError("Invalid loss type")
    
def load_model(model_path) -> NeuralGAM:
    """with open(model_path, "rb") as file:
        return dill.load(file)"""
    import mlflow
    return mlflow.pyfunc.load_model(model_path)


def apply_link(link, a):
    if link == "logistic":
        #return np.where(a >= 0, 1 / (1 + np.exp(-a)), np.exp(a) / (1 + np.exp(a)))
        return np.exp(a) / (1 + np.exp(a))
    else:   # identity / linear
        return a
    
def inv_link(link, a):
    """ Computes the inverse of the link function """ 
    if link == "logistic":
        if type(a) is np.ndarray:
            a = deepcopy(a).astype("float64")
            a[a == 0] += .01 # edge case for log link, inverse link, and logit link
            a[a == 1] -= .01 # edge case for logit link
        return np.log(a) - np.log(1 - a)
    else:   # identity / linear
        return a