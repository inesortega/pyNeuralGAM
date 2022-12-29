from typing import Union
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.python.training.tracking.data_structures import ListWrapper

TfInput = Union[np.ndarray, tf.Tensor]
import dill
import os
class NeuralGAM(tf.keras.Model):
    """
    Neural Generalized Additive Model.

        num_inputs: number of features or variables in input data
        family: distribution family {gaussian, binomial}
        num_units: Number of hidden units in the hidden layer of each feature network - default 1024
    
    """

    def __init__(self,
                num_inputs,
                family,
                num_units = 1024,
                learning_rate = 0.001,
                **kwargs):
        """Initializes NeuralGAM hyperparameters.

        Args:
            num_inputs: Number of feature inputs in input data.
            num_units: Number of hidden units in first layer of each feature net, or list with sizes of N layers [64, 64, 32]
            **kwargs: Arbitrary keyword arguments. Used for passing the `activation`
            function as well as the `name_scope`.
        """
        super(NeuralGAM, self).__init__("NeuralGAM")
        self._num_inputs = num_inputs
        self._family = family
        self._num_units = num_units
        self._kwargs = kwargs
        self.feature_networks = [None] * self._num_inputs
        self.eta0 = 0
        self.y = None
        self.eta = None
        self.lr = learning_rate
        self.build()

    def build_feature_NN(self, layer_name):
        """ Generates a model to fit a specific feature of the dataset"""
        model = Sequential(name=layer_name)

        # add input layer
        model.add(Dense(1))
        # add hidden layer(s). If self._num_units is a list, add one layer per size in the list
        if type(self._num_units) is ListWrapper:
            for i in range(len(self._num_units)):
                model.add(Dense(self._num_units[i], kernel_initializer='glorot_normal', activation='relu'))
        else:
            # single layer (shallow) NeuralGAM
            model.add(Dense(self._num_units, kernel_initializer='glorot_normal', activation='relu'))
        
        model.add(Dense(1))
        model.compile(loss= "mean_squared_error", optimizer=Adam(learning_rate=self.lr))
        return model

    def build(self):
        """Builds a FeatureNNs for each feature """
        for i in range(self._num_inputs):
            self.feature_networks[i] = self.build_feature_NN(layer_name="layer_{0}".format(i))

    def fit(self, X_train, y_train, max_iter_ls, w_train = None, bf_threshold=0.00001, ls_threshold = 0.01, max_iter_backfitting = 10):
        """
        Iteratively fits a NAM model using one DNN per feature in
            
        Parameters: 
            X_train: training samples
            y_train: target training value ({0,1} for binomial family, regression values for gaussian family)
            w_train: (optional) training weights
            ls_threshold: (optional) threshold to stop Local Scoring algorithm. Defaults to 0.01
            bf_threshold: (optional) threshold to stop Backfitting algorithm. Defaults to 0.01
            max_iter: maximum number of iterations of the Local Scoring algorithm (for binomial family only)
            max_iter_backfitting: (optional) maximum number of iterations of the Backfitting algorithm. Defaults to 10
        Returns:
            y: learnt estimator
            g: learned functions for each variable
        """
        print("\n\nFitting GAM \n -- max_it = {0}\n -- max_iter_backfitting = {1}\n -- ls_threshold = {2}\n -- bf_threshold={3}\n -- learning_rate={4}\n\n".format(max_iter_ls, max_iter_backfitting, ls_threshold, bf_threshold, self.lr))
        
        #Initialization
        converged = False
        f = X_train*0
        g = f
        index = f.columns.values
        it = 1
        self.training_err = list()
        if not w_train:
            w = np.ones(len(y_train))   #input weights.... different from Local Scoring Weights!!
        
        if self._family == "gaussian":
            max_iter_ls = 1  # for gaussian, only one iteration of the LS is required!   
        
        muhat = y_train.mean()
        self.eta0 = self.inv_link(muhat)
        eta = self.eta0 #initially estimate eta as the mean of y_train
        dev_new = self.deviance(muhat, y_train, w)
        
        # Start local scoring algorithm
        while (not converged and it <= max_iter_ls):
            
            print("Iter Local Scoring {0}".format(it))
            
            if self._family == "gaussian":
                Z = y_train
                W = w
            else:
                der = self.deriv(muhat)
                Z = eta + (y_train - muhat) * der
                W = self.weight(w, muhat)
            
            it_backfitting = 1
            self.eta0 = Z.mean()
            eta = self.eta0
            eta_prev = self.eta0
            
            # Start backfitting algorithm for max_iter_backfitting iterations inside a given LS loop
            err = bf_threshold + 0.1     
            while( (err > bf_threshold) and (it_backfitting <= max_iter_backfitting)):
                # estimate each function
                for k in range(len(X_train.columns)):
                    
                    #Remove from Z the contributions of other features
                    eta = eta - g[index[k]]
                    residuals = Z - eta

                    if self._family == "binomial":
                        self.feature_networks[k].compile(loss= "mean_squared_error", 
                                                        optimizer="adam",
                                                        loss_weights= pd.Series(W))
                    self.feature_networks[k].fit(X_train[X_train.columns[k]], 
                                                 residuals, 
                                                 epochs=1, 
                                                 sample_weight = pd.Series(W) if self._family == "binomial" else None)
                    # Update f with current learned function for predictor k
                    f[index[k]] = self.feature_networks[k].predict(X_train[X_train.columns[k]])
                    f[index[k]] = f[index[k]] - np.mean(f[index[k]])
                    eta = eta + f[index[k]]  

                # update current estimations
                g = f
                eta = self.eta0 + g.sum(axis=1)
      
                #compute the differences in the predictor at each iteration
                err = np.sum(eta - eta_prev)**2 / np.sum(eta_prev**2)
                eta_prev = eta
                print("BACKFITTING ITERATION #{0}: Current err = {1}".format(it_backfitting, err))
                it_backfitting = it_backfitting + 1
                self.training_err.append(err)

            muhat = self.apply_link(eta)
            dev_old = dev_new
            dev_new = self.deviance(muhat, y_train, w)
            dev_delta = np.abs((dev_old - dev_new)/dev_old)
            
            print("Dev delta = {0}".format(dev_delta))
            if dev_delta < ls_threshold:
                print("Z and f(x) converged...")
                converged = True
            
            it+=1
        
        #  out local scoring
        print("END Local Scoring Algorithm at iteration {0}, dev_delta = {1}, eta0 = {2}".format(it, dev_delta, self.eta0))
        
        # Reconstruct learnt y
        self.y = self.apply_link(eta)
        self.eta = eta

        return self.y, g, eta
    
    def deviance(self, fit, y, W):
        """ Obtains the deviance of the model"""
        if self._family == "gaussian":
            dev = ((y - fit)**2).mean()
        
        elif self._family == "binomial":
            fit = np.where(fit < 0.0001, 0.0001, fit)
            fit = np.where(fit > 0.9999, 0.9999, fit)
            entrop = np.where((1 - y) * y > 0, 
                             2 * (y * np.log(y)) + ((1 - y) * np.log(1 - y)),
                             0)
            entadd = 2 * (y * np.log(fit)) + ( (1-y) * np.log(1-fit))
            dev = np.sum(entrop - entadd)
        return dev 
    
    def deriv(self, muhat):
        """ Computes the derivative of the link function"""
        if self._family == "gaussian":
            out = 1
        
        elif self._family == "binomial":
            prob = muhat
            prob = np.where(prob >= 0.999, 0.999, prob)
            prob = np.where(prob <= 0.001, 0.001, prob)
            prob = prob * (1.0 - prob)
            out = 1.0/prob
            
        elif self._family == "poisson":
            prob = muhat
            prob = np.where(prob <= 0.001, 0.001, prob)
            out = 1.0/prob
            
        return out
    
    def weight(self, w, muhat):
        """Calculates the weights for the Local Scoring"""
        
        if self._family == "gaussian": # Identity
            wei = w
        
        elif self._family == "binomial": # Derivative Logit
            muhat = np.where(muhat <= 0.001, 0.001, muhat)
            muhat = np.where(muhat >= 0.999, 0.999, muhat)
            temp = self.deriv(muhat)
            aux = muhat * (1 - muhat) * (temp**2)
            aux = np.where(aux <= 0.001, 0.001, aux)
            wei = w/aux
        elif self._family == "poisson":
            wei = np.zeros(len(muhat))
            np.where(muhat > 0.01, w/(muhat * self.deriv(muhat)**2), muhat)
        return(wei)
            
    def get_partial_dependencies(self, X: pd.DataFrame):
        output = pd.DataFrame(columns=range(len(X.columns)))
        for i in range(len(X.columns)):
            output[i] = pd.Series(self.feature_networks[i].predict(X[X.columns[i]]).flatten())
        return output
            
    def predict(self, X: pd.DataFrame):
        """Computes Neural GAM output by computing a linear combination of the outputs of individual feature networks."""
        eta = self.get_partial_dependencies(X)
        eta = eta.sum(axis=1) + self.eta0
        y = self.apply_link(eta)
        return y, eta
    
    def apply_link(self, muhat):
        """ Applies the link function """ 
        if self._family == "binomial":
            muhat = np.where(muhat > 10, 10, muhat)
            muhat = np.where(muhat < -10, -10, muhat)
            return np.exp(muhat) / (1 + np.exp(muhat))
        elif self._family == "gaussian":   # identity / gaussian
            return muhat
        elif self._family == "poisson": 
            muhat = np.where(muhat > 300, 300, muhat)
            return np.exp(muhat)
        
    def inv_link(self, muhat):
        """ Computes the inverse of the link function """ 
        if self._family == "binomial":
            d = 1 - muhat 
            d = np.where(muhat <= 0.001, 0.001, muhat)
            d = np.where(muhat >= 0.999, 0.999, muhat)
            return np.log(muhat/d) 
        elif self._family == "gaussian":   # identity / gaussian
            return muhat
        elif self._family == "poisson":
            muhat = np.where(muhat <= 0.001, 0.001, muhat)
            return np.log(muhat)
    
    def compute_err(self, actual, pred):
        """ Compute err according to the distribution family"""
        if self._family == "binomial":
            # calculate binary cross entropy
            sum_score = 0.0
            for i in range(len(actual)):
                sum_score += actual[i] * np.log(1e-15 + pred[i])
            mean_sum_score = 1.0 / len(actual) * sum_score
            return -mean_sum_score
        elif self._family == "gaussian":
            return mean_squared_error(actual, pred)    
        elif self._family == "multinomial":
            # calculate categorical cross entropy
            sum_score = 0.0
            for i in range(len(actual)):
                for j in range(len(actual[i])):
                    sum_score += actual[i][j] * np.log(1e-15 + pred[i][j])
            mean_sum_score = 1.0 / len(actual) * sum_score
            return -mean_sum_score
        else:
            raise ValueError("Invalid family type")

    
def plot_partial_dependencies(x: pd.DataFrame, fs: pd.DataFrame, title: str, output_path: str = None):    
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn')

    params = {"axes.linewidth": 2,
            "font.family": "serif",
            'font.size': 20}

    axis_font = {'fontname':'serif', 'size':'14'}
    matplotlib.rcParams['agg.path.chunksize'] = 10000
    plt.rcParams.update(params)
    plt.style.use('seaborn-darkgrid')

    fig, axs = plt.subplots(nrows=1, ncols=len(fs.columns), figsize=(25,20))
    fig.suptitle(title)
    
    for i, term in enumerate(fs.columns):
        data = pd.DataFrame()
        data['x'] = x[x.columns[i]]
        data['f(x)']= fs[fs.columns[i]]
        
        sns.lineplot(data = data, x='x', y='f(x)', color='royalblue', ax=axs[i])
        axs[i].grid()
        axs[i].set_xlabel(f'$X_{i+1}$', **axis_font)
        axs[i].set_ylabel(f'$f(x_{i+1})$', **axis_font)

        # Set the tick labels font
        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            label.set_fontname('serif')
            label.set_fontsize(20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi = 300, bbox_inches = "tight")
        fig = plt.gcf()