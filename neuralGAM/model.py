from concurrent.futures import ThreadPoolExecutor
import logging
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
import logging

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
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning, 
    message="divide by zero encountered in log"
)
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning, 
    message="invalid value encountered in multiply"
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class NeuralGAM(tf.keras.Model):
    """
    Neural Generalized Additive Model with parametric and nonparametric components.
    
    For the nonparametric part, one neural network is built per feature.
    """
    def __init__(self, family, num_units=1024, learning_rate=0.001,
                 activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=None, linear_terms = None, verbose = 1, **kwargs):
        """
        Initialize the NeuralGAM model.
        Parameters:
            family (str): The family of distributions (e.g., 'gaussian', 'binomial').
            num_units (int or list, optional): Number of units in the hidden layers. Default is 1024.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
            activation (str, optional): Activation function to use. Default is 'relu'.
            kernel_initializer (str, optional): Initializer for the kernel weights matrix. Default is 'glorot_normal'.
            kernel_regularizer (optional): Regularizer function applied to the kernel weights matrix. Default is None.
            linear_terms (list, optional): List of linear terms. Default is None.
            verbose (int, optional): Verbosity mode. Default is 1.
            **kwargs: Additional keyword arguments.
        Attributes:
            family (str): The family of distributions.
            p_terms (list): List of linear terms.
            np_terms (list): List of non-parametric terms. Generated from the input data - the linear terms defined in p_terms
            num_units (int or list): Number of units in the hidden layers.
            lr (float): Learning rate for the optimizer.
            activation (str): Activation function to use.
            kernel_initializer (str): Initializer for the kernel weights matrix.
            kernel_regularizer: Regularizer function applied to the kernel weights matrix.
            verbose (int): Verbosity mode.
            eta0: Overall intercept.
            y: Final fitted response.
            eta: Final additive predictor.
            feature_contributions: Fitted feature contributions.
            training_err (list): List to track backfitting convergence.
        """
        super(NeuralGAM, self).__init__()
        
        self.family = family.lower()
        self.p_terms = list(linear_terms) if linear_terms is not None else []
        self.np_terms = list()
        self.num_units = num_units if isinstance(num_units, int) else list(num_units)
        self.lr = learning_rate
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.verbose = verbose
        # Placeholders for the learned components:
        self.eta0 = None  # overall intercept
        self.y = None     # final fitted response
        self.eta = None   # final additive predictor
        self.learned_fs = None # learned functions
        self.training_err = []  # to track backfitting convergence
        
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
                        sample_weight=pd.Series(W) if self.family == "binomial" else None, verbose=self.verbose)

        f_k = self.feature_networks[term].predict(X_train[[term]], verbose = self.verbose)
        f_k = f_k - np.mean(f_k)
        return f_k

    def fit(self, X_train, y_train, max_iter_ls=10, w_train=None,
            bf_threshold=0.001, ls_threshold=0.1, max_iter_backfitting=10, parallel=True):
        """
        Fit the Generalized Additive Model (GAM) to the training data.
        Parameters:
        -----------
            X_train : pandas.DataFrame
            The input features for training.
            y_train : pandas.Series or numpy.ndarray
            The target values for training.
            max_iter_ls : int, optional (default=10)
            Maximum number of iterations for the local scoring loop.
            w_train : numpy.ndarray, optional (default=None)
            Weights for the training samples.
            bf_threshold : float, optional (default=0.001)
            Threshold for convergence in the backfitting loop.
            ls_threshold : float, optional (default=0.1)
            Threshold for convergence in the local scoring loop.
            max_iter_backfitting : int, optional (default=10)
            Maximum number of iterations for the backfitting loop.
            parallel : bool, optional (default=True)
            Whether to use parallel execution for backfitting.
        Returns:
        --------
        self : object. The fitted model.
            - self.feature_contributions: fitted feature contributions
            - self.y: fitted response variable
            - self.eta: additive predictor
        """
        if len(self.p_terms) > 0:
            self.np_terms = list(X_train.columns)
            # remove from self.np_terms the linear terms
            self.np_terms = [x for x in self.np_terms if x not in self.p_terms]
        else:
            self.np_terms = list(X_train.columns)
        
        self.parametric_model = None  # Linear model for parametric terms
        self.feature_networks = {term: None for term in self.np_terms}   # one NN per nonparametric feature
        self.build_networks()


        logger.debug(f"Fitting GAM")
        logger.debug(f" -- local scoring iter = {max_iter_ls}")
        logger.debug(f" -- backfitting iter = {max_iter_backfitting}")
        logger.debug(f" -- ls_threshold = {ls_threshold}")
        logger.debug(f" -- bf_threshold = {bf_threshold}")
        logger.debug(f" -- learning_rate = {self.lr}")
        
        
        if parallel:
            logger.debug("Using parallel execution")

        else:
            logger.debug("Using sequential execution")

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
            logger.info(f"Local Scoring Iteration - {it}")
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
                logger.info(f"BACKFITTING ITERATION #{it_backfitting}: Current err = {err}")
                it_backfitting = it_backfitting + 1
                self.training_err.append(err)

            muhat = self.apply_link(eta)
            dev_old = dev_new
            dev_new = self.deviance(muhat, y_train, w)
            dev_delta = np.abs((dev_old - dev_new) / dev_old)
            logger.info(f"Dev delta = {dev_delta}")
            if dev_delta < ls_threshold:
                logger.info("Convergence achieved.")
                converged = True
            it += 1
        
        # Final fitted response.
        self.y = self.apply_link(eta)
        self.eta = eta
        self.feature_contributions = g
        return self

    def deviance(self, fit, y, W):
        """
        Calculate the deviance for the given model fit, observed values, and weights.
        Parameters:
        fit (array-like): The predicted values from the model.
        y (array-like): The observed values.
        W (array-like): The weights for each observation.
        Returns:
        float: The deviance of the model fit.
        """
        
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
        """
        Computes the derivative of the link function based on the specified family.
        Parameters:
        muhat (array-like): The predicted mean values.
        Returns:
        array-like: The derivative of the link function.
        Raises:
        ValueError: If the family is not one of 'gaussian', 'binomial', or 'poisson'.
        Notes:
        - For the 'gaussian' family, the derivative is always 1.
        - For the 'binomial' family, the derivative is computed as 1 / (prob * (1 - prob)),
          where prob is clipped to the range [0.001, 0.999] to avoid division by zero.
        - For the 'poisson' family, the derivative is computed as 1 / prob,
          where prob is clipped to a minimum value of 0.001 to avoid division by zero.
        """
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
        """
        Calculates the weights for the Local Scoring algorithm based on the specified family.
        Parameters:
        w (array-like): Initial weights.
        muhat (array-like): Estimated mean values.
        Returns:
        array-like: Calculated weights for the Local Scoring.
        Notes:
        - For the "gaussian" family, the weights are returned as is.
        - For the "binomial" family, the weights are adjusted using the derivative of the logit function.
        - For the "poisson" family, the weights are adjusted based on the estimated mean values and their derivatives.
        """
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
        """
        Applies the link function to the given input based on the specified family.

        Parameters:
        -----------
        muhat : array-like
            The input values to which the link function is applied.

        Returns:
        --------
        array-like
            The transformed values after applying the link function.

        Raises:
        -------
        ValueError
            If the family attribute is not one of "binomial", "gaussian", or "poisson".

        Notes:
        ------
        - For the "binomial" family, the logit link function is applied with clipping to avoid overflow.
        - For the "gaussian" family, the identity link function is applied.
        - For the "poisson" family, the log link function is applied with clipping to avoid overflow.
        """
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
        """
        Computes the inverse of the link function based on the specified family.

        Parameters:
        -----------
        muhat : array-like
            The predicted mean values.

        Returns:
        --------
        array-like
            The transformed values after applying the inverse link function.

        Notes:
        ------
        - For the "binomial" family, the inverse link function is the logit function.
        - For the "gaussian" family, the inverse link function is the identity function.
        - For the "poisson" family, the inverse link function is the natural logarithm.
        - Values of `muhat` are clipped to avoid numerical issues:
            - For "binomial", `muhat` is clipped to the range [0.001, 0.999].
            - For "poisson", `muhat` is clipped to a minimum of 0.001.
        """
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
        elif self.family == "multinomial":
            d = 1 - muhat 
            d = np.where(muhat <= 0.001, 0.001, muhat)
            d = np.where(muhat >= 0.999, 0.999, muhat)
            # apply the softmax function
            exp_muhat = np.exp(muhat - np.max(muhat, axis=1))
            return exp_muhat / np.sum(exp_muhat, axis=1)
            
    def predict(self, X, type="link", terms=None, verbose=1):
        """
        Predicts the output for the given input data.
        Parameters:
        -----------
        X : pd.DataFrame
            Input data for prediction. Must be a pandas DataFrame.
        type : str, optional (default="link")
            Type of prediction to return. Valid options are:
            - "link": Returns the linear predictor (eta).
            - "terms": Returns the predictions for each term.
            - "response": Returns the response variable after applying the link function.
        terms : list of str, optional
            Specific terms to include in the prediction when type is "terms". If None, all terms are included.
        verbose : int, optional (default=1)
            Verbosity level.
        Returns:
        --------
        pd.Series or pd.DataFrame
            The predicted values. The return type depends on the `type` parameter:
            - "link": Returns a pd.Series with the linear predictor (eta).
            - "terms": Returns a pd.DataFrame with predictions for each term.
            - "response": Returns a pd.Series with the response variable.
        Raises:
        -------
        ValueError
            If `X` is not a pandas DataFrame.
            If `type` is not one of the valid options.
            If `terms` contains invalid column names.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        
        def get_model_predictions(X, term):
            """Helper function to get model predictions for a specific term."""
            if term in self.np_terms:
                return pd.Series(self.feature_networks[term].predict(X[term], verbose = self.verbose).flatten())
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