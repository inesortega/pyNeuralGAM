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
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
        
class NeuralGAM(tf.keras.Model):
    """
    Neural Generalized Additive Model.
    """

    def __init__(self,
                num_inputs,
                family,
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
        self.family = family
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
        model.compile(loss= "mean_squared_error", optimizer="adam", metrics=['mean_absolute_error'])
        
        return model


    def build(self):
        """Builds a FeatureNNs for each feature """
        for i in range(self._num_inputs):
            self.feature_networks[i] = self.build_feature_NN()

    def fit(self, X_train, y_train, max_iter, convergence_threshold, w_train = None):
        
        #Initialization
        print("Fitting GAM - max_it = {0} ; convergence_thresold = {1}".format(max_iter, convergence_threshold))
        
        converged = False
        f = X_train*0
        g = f
        index = f.columns.values
        DELTA_THRESHOLD = 0.001
        it = 0
        
        it_backfitting = 1
        err = convergence_threshold + 0.1
        max_iter_backfitting = 3
        
        if not w_train:
            w = np.ones(len(y_train))   #input weights.... different from Local Scoring Weights
        
        if self.family == "gaussian":
            max_iter = 1
        
        self.muhat = y_train.mean()
        eta = inv_link(self.family, self.muhat)
        dev_new = self.deviance(self.muhat, y_train, w)
        
        # Start local scoring algorithm
        while (not converged and (it <= max_iter)):
            
            it_backfitting = 1
            
            print("Iter Local Scoring = {0}".format(it))
            
            if self.family == "gaussian":
                Z = y_train
                W = w
            else:
                der = self.deriv(self.muhat, self.family)
                Z = eta + (y_train - self.muhat) * der
                W = self.weight(w, self.muhat, self.family)
            
            # Start backfitting algorithm
            while( (err > convergence_threshold) and (it_backfitting <= max_iter_backfitting)):
                
                for k in range(len(X_train.columns)):
                    idk = (list(range(0,k,1))+list(range(k+1,len(X_train.columns),1)))    # Get idx of columns != k
                    
                    # Compute the partial residual - remove from y the contribution from other features
                    residuals = Z - Z.mean() - g[index[idk]].sum(axis=1)
                    
                    # Fit network k with X_train[k] towards residuals
                    #self.feature_networks[k].fit(X_train[X_train.columns[k]]/W, residuals, epochs=1) 
                    
                    polynomial = PolynomialFeatures(degree=2).fit_transform(X_train[X_train.columns[k]].to_numpy().reshape(-1,1))
                    model = linear_model.LinearRegression()
                    model.fit(polynomial, residuals, W)
                    
                    # Update f with current learned function for predictor k -- get f ready for compute residuals at next iteration
                    #f[index[k]] = self.feature_networks[k].predict(X_train[X_train.columns[k]])
                    f[index[k]] = model.predict(polynomial)
                    f[index[k]] = f[index[k]] - np.mean(f[index[k]])  
                
                # update current estimations
                g = f
                eta_prev = eta
                eta = Z.mean() + g.sum(axis=1)
      
                #compute the differences in the predictor at each iter
                err = np.sum(np.abs(eta_prev - eta)) / np.sum(np.abs(eta_prev))
                print("ITERATION #{0}: Current err = {1}".format(it_backfitting, err))
                it_backfitting += 1
                self.training_mse.append(err)

            self.muhat = apply_link(self.family, eta)
            dev_old = dev_new
            dev_new = self.deviance(self.muhat, y_train, w)
            dev_delta = np.abs((dev_old - dev_new)/dev_old)
            if dev_delta < DELTA_THRESHOLD:
                print("Z and f(x) converged...")
                converged = True
            
            it+=1
        
        #  out local scoring
        """mus = g.mean(axis=0)
        g = g - mus
        eta0 = eta0 + mus
        eta = (eta0 + g).sum(axis=1)"""
        self.y = apply_link(self.family, eta)
        
        return self.y, g
    
    def deviance(self, fit, y, W):
        """ obtains the deviance of the model"""
        if self.family == "gaussian":
            dev = ((y - fit)**2).mean()
        
        elif self.family == "binomial":
            fit = np.where(fit < 0.0001, 0.0001, fit)
            fit = np.where(fit > 0.9999, 0.9999, fit)
            
            entrop = np.zeros(len(y))
            ii = np.where((1 - y) * y > 0)
            indexes = ii[0] # get indexes of y where (1 - y) * y > 0
            if len(indexes) > 0:
                entrop[indexes] = 2 * (y[indexes] * np.log(y[indexes])) + ((1 - y[indexes]) * np.log(1 - y[indexes]))
            entadd = 2 * (y * np.log(fit)) + ( (1-y) * np.log(1-fit))
            dev = np.sum(entrop - entadd)
        return dev 
    
    def deriv(self, muhat, family):
        """ Computes the derivative of the link function"""
        if family == "gaussian":
            out = 1
        
        elif family == "binomial":
            prob = muhat
            prob = np.where(prob >= 0.999, 0.999, prob)
            prob = np.where(prob <= 0.001, 0.001, prob)
            prob = prob * (1.0 - prob)
            out = 1.0/prob
            
        elif family == "poisson":
            prob = muhat
            prob = np.where(prob <= 0.001, 0.001, prob)
            out = 1.0/prob
            
        return out
    
    def weight(self, w, muhat, family):
        """Calculates the weights for the Local Scoring"""
        
        if family == "gaussian": # Identity
            wei = w
        
        elif family == "binomial": # Derivative Logit
            muhat = np.where(muhat <= 0.001, 0.001, muhat)
            muhat = np.where(muhat >= 0.999, 0.999, muhat)
            temp = self.deriv(muhat, family)
            aux = muhat * (1 - muhat) * (temp**2)
            aux = np.where(aux <= 0.001, 0.001, aux)
            wei = w/aux
        elif family == "poisson":
            wei = np.zeros(len(muhat))
            np.where(muhat > 0.01, w/(muhat * self.deriv(muhat, family)**2), muhat)
        return(wei)
            
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
            output = output.apply(lambda x: apply_link(self.family, x))
        return output
            
    def predict(self, X: pd.DataFrame):
        """Computes Neural GAM output by computing a linear combination of the outputs of individual feature networks."""
        output = self.get_partial_dependencies(X, xform=False)
        y = apply_link(self.family, output.sum(axis=1) + self.muhat)
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
    """ 
    with open(model_path, "rb") as file:
        return dill.load(file) 
    """
    import mlflow
    return mlflow.pyfunc.load_model(model_path)


def apply_link(family, muhat):
    if family == "binomial":
        muhat = np.where(muhat > 10, 10, muhat)
        muhat = np.where(muhat < -10, -10, muhat)
        return np.exp(muhat) / (1 + np.exp(muhat))
    elif family == "gaussian":   # identity / gaussian
        return muhat
    elif family == "poisson":   # identity / gaussian
        muhat = np.where(muhat > 300, 300, muhat)
        return np.exp(muhat)
    
def inv_link(family, muhat):
    """ Computes the inverse of the link function """ 
    if family == "binomial":
        d = 1 - muhat 
        d = np.where(muhat <= 0.001, 0.001, muhat)
        d = np.where(muhat >= 0.999, 0.999, muhat)
        return np.log(muhat/d) 
    elif family == "gaussian":   # identity / gaussian
        return muhat
    elif family == "poisson":
        muhat = np.where(muhat <= 0.001, 0.001, muhat)
        return np.log(muhat)
        