
""" This file contains a set of auxiliar functions to plot and compute results, and generate synthetic datasets with different distributions """

import itertools
import os
from sklearn.metrics import roc_curve
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from scipy.stats import truncnorm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')

params = {"axes.linewidth": 2,
        "font.family": "serif"}

matplotlib.rcParams['agg.path.chunksize'] = 10000
plt.rcParams.update(params)

def split(X, y, fs, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    
    fs_train = fs.iloc[X_train.index].reset_index(drop=True)
    fs_test = fs.iloc[X_test.index].reset_index(drop=True)

    print(f"Shape train {y_train.shape}")
    print(f"Shape test {y_test.shape}")

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True).squeeze(), y_test.reset_index(drop=True).squeeze(), fs_train, fs_test

def save(X_train,X_test,y_train,y_test, output_folder):
    import os
    output_path = os.path.normpath(os.path.abspath(output_folder))
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    X_train.to_csv(os.path.join(output_path, "x_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_path, "x_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)
        
""" DATA GENERATION """

def generate_err(nrows:int, err_type:str, eta0:pd.DataFrame):
    err = np.random.normal(loc=0, scale=0.5, size=nrows)
    if err_type == "heteroscedastic":
        sigma = 0.5 + np.abs(0.25*eta0)
        err = err * sigma

    print("\n Intercept: {0} data".format(err_type))
    print(pd.DataFrame(err).describe())

    return err

def get_truncated_normal(mean=0, sd=1, low=0, upp=10, nrows=25000):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(nrows)
    
def generate_normal_data(nrows, err_type, family):
    """
    Generates a dataset with normally distributed features and a target variable.
    Parameters:
        nrows (int): Number of rows (samples) to generate.
        err_type (str): Type of error to introduce in the target variable {homoscedastic, heteroscedastic}.
        family (str): Family of the distribution for the target variable.
        seed (int, optional): Seed for the random number generator. Defaults to 343142.
    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): DataFrame with the generated features (x1, x2, x3).
            - y (pd.Series): Series with the generated target variable.
            - fs (pd.DataFrame): DataFrame with the transformed features used to compute the target variable.
    """
    np.random.seed(seed)
    x1 = get_truncated_normal(mean=0.0, sd=1.0, low=-5, upp=5, nrows=nrows)
    x2 = get_truncated_normal(mean=0.0, sd=1.0, low=-5, upp=5, nrows=nrows)
    x3 = get_truncated_normal(mean=0.0, sd=1.0, low=-5, upp=5, nrows=nrows)
    beta0 = np.ones(nrows) * 2
    
    X = pd.DataFrame([x1,x2,x3]).transpose()
    fs = pd.DataFrame([x1*x1, 2*x2, np.sin(x3)]).transpose()
    print("y = beta0 + f(x1) + f(x2) + f(x3) =  2 + x1^2 + 2x2 + sin(x3)")
   
    y = compute_y(fs, beta0, nrows, err_type, family)
    
    return X, y, fs

def generate_uniform_data(nrows, err_type, family):
    """
    Generates uniform data for a specified number of rows and computes the response variable.
    Parameters:
    nrows (int): Number of rows of data to generate.
    err_type (str): Type of error to introduce in the target variable {homoscedastic, heteroscedastic}.
    family (str): Family of the distribution for the response variable.
    Returns:
    tuple: A tuple containing:
        - X (pd.DataFrame): DataFrame containing the generated features x1, x2, and x3.
        - y (np.ndarray): Array containing the computed response variable.
        - fs (pd.DataFrame): DataFrame containing the functions of the features.
    """
    if family != "gaussian" and family != "binomial":
        raise ValueError("Family must be either 'gaussian' or 'binomial'")
    x1 = np.array(np.random.uniform(low=-2.5, high=2.5, size=nrows))
    x2 = np.array(np.random.uniform(low=-2.5, high=2.5, size=nrows))
    x3 = np.array(np.random.uniform(low=-2.5, high=2.5, size=nrows))
    beta0 = np.ones(nrows) * 2 
    X = pd.DataFrame([x1,x2,x3]).transpose()
    fs = pd.DataFrame([x1*x1, 2*x2, np.sin(x3)]).transpose()
    print("y = beta0 + f(x1) + f(x2) + f(x3) =  2 + x1^2 + 2x2 + sin(x3)")   
    y = compute_y(fs, beta0, nrows, err_type, family)
    
    return X, y, fs

def compute_y(fs, beta0, nrows, err_type, family):
    
    y = fs.sum(axis=1) + beta0
    
    if family == "binomial":
        y = y - np.mean(y)
        y = np.exp(y)/(1+np.exp(y)) # Probabilities of success       
        
    elif family == "gaussian":
        err = generate_err(nrows=nrows, err_type=err_type, eta0=y)
        y = y + err
        y = y - np.mean(y)
    return pd.Series(y)

def generate_data(err_type, distribution, family, nrows=25000, seed = 343142):
    """
        Returns a pair of X,y to be used with NeuralGAM
        :param: err_type: homogeneity of variance on the intercept term {homoscedastic, heteroscedastic}
        :param: distribution: generate normal or uniform distributed X data {uniform, normal}
        :param: family: generate reponse Y for linear or binomial regression problems
        :param: nrows: data size (number of rows)
        :return: X: pandas Dataframe object with generated X (one column per feature). Xs follow a normal distribution
        :return: y: pandas Series object with computed y, with a normal distribution + homoskedastic residual
    """
    if distribution != "uniform" and distribution != "normal":
        raise ValueError("Distribution must be either 'uniform' or 'normal'")
    if family != "gaussian" and family != "binomial":
        raise ValueError("Family must be either 'gaussian' or 'binomial'")
    if nrows <= 0:
        raise ValueError("Number of rows must be greater than 0")
    
    np.random.seed(seed)

    if distribution == "uniform":
        X, y, fs = generate_uniform_data(nrows, err_type, family)

    elif distribution == "normal":
        X, y, fs = generate_normal_data(nrows, err_type, family)   

    return X, y, fs