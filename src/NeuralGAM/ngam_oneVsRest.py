from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
if __debug__:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ['CUDA_VISIBLE_DEVICES'] = "" ### Disable GPU

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression



warnings.simplefilter(action='ignore', category=FutureWarning)

class NeuralGAM(tf.keras.Model):
    """
    Neural Generalized Additive Model with parametric and nonparametric components.
    
    For the nonparametric part, one neural network is built per feature.
    """
    def __init__(self, p_terms, np_terms, family, num_units=1024, learning_rate=0.001,
                 activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=None,
                 loss=None, **kwargs):
        super(NeuralGAM, self).__init__()
        
        self.family = family.lower()
        self.p_terms = list(p_terms) if p_terms is not None else []
        self.np_terms = list(np_terms) if np_terms is not None else []
        self._num_inputs = len(self.np_terms)
        self.num_units = num_units if isinstance(num_units, int) else list(num_units)
        self.lr = learning_rate
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.loss = loss
        # Placeholders for the learned components:
        self.parametric_model = None  # Linear model for parametric terms
        self.feature_networks = {term: None for term in self.np_terms}   # one NN per nonparametric feature
        self.eta0 = None  # overall intercept
        self.y = None     # final fitted response
        self.eta = None   # final additive predictor
        self.training_err = []  # to track backfitting convergence
        
        self.build_networks()

    def build_feature_NN(self, layer_name):
        """Builds the neural network for one nonparametric term."""
        model = Sequential(name=layer_name)
        # Input layer for 1D predictor:
        model.add(Dense(1, input_shape=(1,)))
        # Hidden layers:
        if isinstance(self.num_units, int):
            model.add(Dense(self.num_units, kernel_initializer=self.kernel_initializer,
                            activation=self.activation))
        else:
            for units in self.num_units:
                model.add(Dense(units, kernel_initializer=self.kernel_initializer,
                                activation=self.activation))
        # Output layer: one output unit
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.lr))
        return model

    def build_networks(self):
        """Initialize a neural network for every nonparametric term."""
        for term in self.np_terms:
            self.feature_networks[term] = self.build_feature_NN(layer_name=f"layer_{term}")

    def process_feature(self, term, eta, g, W, Z, X_train):
        """
        Update one nonparametric component:
          - Remove its current contribution from eta.
          - Compute the residual.
          - Train the corresponding NN on that predictor.
        """
        eta = eta - g[term]
        residuals = Z - eta

        if self.family == "binomial":
            self.feature_networks[term].compile(loss="mean_squared_error", 
                                    optimizer="adam",
                                    loss_weights=pd.Series(W))
        self.feature_networks[term].fit(X_train[[term]], 
                        residuals, 
                        epochs=1, 
                        sample_weight=pd.Series(W) if self.family == "binomial" else None)

        f_k = self.feature_networks[term].predict(X_train[[term]])
        f_k = f_k - np.mean(f_k)
        return f_k

    def fit(self, X_train, y_train, max_iter_ls=10, w_train=None,
            bf_threshold=0.001, ls_threshold=0.1, max_iter_backfitting=10, parallel = True):
        """
        Fit the NeuralGAM model using local scoring and backfitting.
        
        X_train: DataFrame with all predictors (both parametric and nonparametric).
        y_train: Response vector.
        """
        print("\nFitting GAM")
        print(f" -- local scoring iter = {max_iter_ls}")
        print(f" -- backfitting iter = {max_iter_backfitting}")
        print(f" -- ls_threshold = {ls_threshold}")
        print(f" -- bf_threshold = {bf_threshold}")
        print(f" -- learning_rate = {self.lr}\n")
        
        
        if parallel:
            print("Using parallel execution")

        else:
            print("Using sequential execution")

        converged = False
        f = X_train*0
        g = X_train*0
        it = 1

        # For gaussian, only one local scoring iteration is used.
        if self.family == "gaussian":
            max_iter_ls = 1
        
        self.training_err = list()
        if not w_train:
            w = np.ones(len(y_train))   #input weights.... different from Local Scoring Weights!!
        
        if self.family == "gaussian":
            max_iter_ls = 1  # for gaussian, only one iteration of the LS is required!   
        
        muhat = y_train.mean()
        self.eta0 = self.inv_link(muhat)
        eta = self.eta0 #initially estimate eta as the mean of y_train
        dev_new = self.deviance(muhat, y_train, w)
        
        # Local scoring loop.
        while (not converged and it <= max_iter_ls):
            print("Local Scoring Iteration", it)
            if self.family == "gaussian":
                Z = y_train
                W = w
            else:
                der = self.deriv(muhat)
                Z = eta + (y_train - muhat) * der
                W = self.weight(w, muhat)
            
            # Update parametric part if present.
            if len(self.p_terms) > 0:
                param_model = LinearRegression()
                self.parametric_model = param_model.fit(X_train[self.p_terms], Z)
                self.eta0 = param_model.intercept_
                f[self.p_terms] = param_model.predict(X_train[self.p_terms]).reshape(-1,1)
                eta = self.eta0 + f.sum(axis=1)
            else:
                self.eta0 = np.mean(Z)
                eta = self.eta0
            
            eta_prev = eta.copy()
            it_backfitting = 1
            err = bf_threshold + 0.1
            
            # Backfitting loop for nonparametric terms.
            while( (err > bf_threshold) and (it_backfitting <= max_iter_backfitting)):
                
                if parallel:
                    with ThreadPoolExecutor(max_workers=None) as executor:
                        futures = [executor.submit(self.process_feature, term, eta, g, W, Z, X_train) for term in self.np_terms]
                        results = [future.result() for future in futures]
                        
                    # Update f and eta with the results from the parallel execution
                    for k, f_k in enumerate(results):
                        f[self.np_terms[k]] = f_k
                else:
                    for term in self.np_terms:
                        f[term] = self.process_feature(term, eta, g, W, Z, X_train)
                
                # update current estimations
                g = f.copy(deep=True)
                eta = self.eta0 + g.sum(axis=1)
                
                # compute the differences in the predictor at each iteration
                err = np.sum(eta - eta_prev)**2 / np.sum(eta_prev**2)
                eta_prev = eta
                print("BACKFITTING ITERATION #{0}: Current err = {1}".format(it_backfitting, err))
                it_backfitting = it_backfitting + 1
                self.training_err.append(err)

            muhat = self.apply_link(eta)
            dev_old = dev_new
            dev_new = self.deviance(muhat, y_train, w)
            dev_delta = np.abs((dev_old - dev_new) / dev_old)
            print("Dev delta =", dev_delta)
            if dev_delta < ls_threshold:
                print("Convergence achieved.")
                converged = True
            it += 1
        
        # Final fitted response.
        self.y = self.apply_link(eta)
        self.eta = eta
        return self.y, g, eta

    def deviance(self, fit, y, W):
        """ Obtains the deviance of the model"""
        if self.family == "gaussian":
            dev = ((y - fit)**2).mean()
        
        elif self.family == "binomial":
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
        if self.family == "gaussian":
            out = 1
        
        elif self.family == "binomial":
            prob = muhat
            prob = np.where(prob >= 0.999, 0.999, prob)
            prob = np.where(prob <= 0.001, 0.001, prob)
            prob = prob * (1.0 - prob)
            out = 1.0/prob
            
        elif self.family == "poisson":
            prob = muhat
            prob = np.where(prob <= 0.001, 0.001, prob)
            out = 1.0/prob
            
        return out
    
    def weight(self, w, muhat):
        """Calculates the weights for the Local Scoring"""
        
        if self.family == "gaussian": # Identity
            wei = w
        
        elif self.family == "binomial": # Derivative Logit
            muhat = np.where(muhat <= 0.001, 0.001, muhat)
            muhat = np.where(muhat >= 0.999, 0.999, muhat)
            temp = self.deriv(muhat)
            aux = muhat * (1 - muhat) * (temp**2)
            aux = np.where(aux <= 0.001, 0.001, aux)
            wei = w/aux
        elif self.family == "poisson":
            wei = np.zeros(len(muhat))
            np.where(muhat > 0.01, w/(muhat * self.deriv(muhat)**2), muhat)
        return(wei)
    
    def apply_link(self, muhat):
        """ Applies the link function """ 
        if self.family == "binomial":
            muhat = np.where(muhat > 10, 10, muhat)
            muhat = np.where(muhat < -10, -10, muhat)
            return np.exp(muhat) / (1 + np.exp(muhat))
        elif self.family == "gaussian":   # identity / gaussian
            return muhat
        elif self.family == "poisson": 
            muhat = np.where(muhat > 300, 300, muhat)
            return np.exp(muhat)
        
    def inv_link(self, muhat):
        """ Computes the inverse of the link function """ 
        if self.family == "binomial":
            d = 1 - muhat 
            d = np.where(muhat <= 0.001, 0.001, muhat)
            d = np.where(muhat >= 0.999, 0.999, muhat)
            return np.log(muhat/d) 
        elif self.family == "gaussian":   # identity / gaussian
            return muhat
        elif self.family == "poisson":
            muhat = np.where(muhat <= 0.001, 0.001, muhat)
            return np.log(muhat)
            
    def predict(self, X, type="link", terms=None, verbose=1):
        """
        Predict method for NeuralGAM.
        
        X: DataFrame with predictors.
        type: Type of prediction ('link', 'terms', 'response').
            - link: returns the additive predictor.
            - terms: returns the predictions for specific terms.
            - response: returns the final response.
        terms: Specific terms to compute if type is 'terms'.
        verbose: Verbosity level.
        """
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        
        def get_model_predictions(X, term):
            """Helper function to get model predictions for a specific term."""
            if term in self.np_terms:
                return pd.Series(self.feature_networks[term].predict(X[term]).flatten())
            if len(self.p_terms) > 0:
                return pd.Series(self.parametric_model.predict(X[self.p_terms]).flatten())
        
        valid_types = ["link", "terms", "response"]
        if type not in valid_types:
            raise ValueError(f"Invalid type argument. Valid options are {valid_types}.")
        
        if type == "terms" and terms is not None and not all(term in X.columns for term in terms):
            raise ValueError(f"Invalid terms. Valid options are: {', '.join(X.columns)}")
        
        f = pd.DataFrame(0, index=X.index, columns=X.columns)
        
        for term in X.columns:
            if type == "terms" and terms is not None:
                if term in terms:
                    f[term] = get_model_predictions(X, term)
                else:
                    continue
            else:
                f[term] = get_model_predictions(X, term)
        
        if type == "terms":
            if terms is not None:
                f = f[terms]
            return f
        
        eta = f.sum(axis=1) + self.eta0
        if type == "link":
            return eta
        
        if type == "response":
            y = self.apply_link(eta)
            return y

class NeuralGAMMultinomial:
    def __init__(self, p_terms, np_terms, num_classes, num_units=1024, learning_rate=0.001, **kwargs):
        self.num_classes = num_classes
        # Create one NeuralGAM per class using the binomial family
        self.models = [NeuralGAM(p_terms=p_terms, np_terms=np_terms, family="binomial", 
                                   num_units=num_units, learning_rate=learning_rate, **kwargs)
                       for _ in range(num_classes)]
    
    def fit(self, X_train, y_train, **fit_params):
        # Fit each model with binary labels for the corresponding class

        # Check X_train contains p_terms and np_terms:
        for model in self.models:
            assert all(term in X_train.columns for term in model.p_terms + model.np_terms)

        for k in range(self.num_classes):
            # Create binary targets for current class (one-vs-rest ): 1 if the class matches, else 0
            y_binary = (y_train == k).astype(int)
            print(f"Training model for class {k}")
            self.models[k].fit(X_train, y_binary, **fit_params)
    
    def predict(self, X_test):
        # Get predictions from each binary model
        preds = np.column_stack([model.predict(X_test)[0] for model in self.models])
        # Optionally, normalize using softmax to get calibrated probabilities:
        exp_preds = np.exp(preds)
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        # Predicted class is the one with highest probability
        predicted_class = np.argmax(probs, axis=1)
        return predicted_class, probs