import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import scipy
from src.NeuralGAM.ngam import NeuralGAM, load_model

#todo delete in prod!!
np.random.seed = 42
      
""" PLOTTING aux functions """
def plot_list(dataframe_list: list, legends: list, title:str):
    fig, axs = plt.subplots(1, len(dataframe_list))
    fig.suptitle(title, fontsize=16)
    for i, term in enumerate(dataframe_list):
        axs[i].plot(dataframe_list[i])
        axs[i].grid()
        axs[i].set_title(legends[i])

    
def plot_partial_dependencies(x, fs, title:str):
    fig, axs = plt.subplots(nrows=1, ncols=len(fs.columns))
    fig.suptitle(title, fontsize=16)
    for i, term in enumerate(fs.columns):
        
        data = pd.DataFrame()
        
        data['x'] = x[x.columns[i]]
        data['y']= fs[fs.columns[i]]
        # calculate confidence interval at 95%
        ci = 1.96 * np.std(data['y'])/np.sqrt(len(data['x']))
        
        data['y+ci'] = data['y'] + ci
        data['y-ci'] = data['y'] - ci
        sns.lineplot(data = data, x='x', y='y', color='b', ax=axs[i])
        sns.lineplot(data = data, x='x', y='y-ci', color='r', linestyle='--', ax=axs[i])
        sns.lineplot(data = data, x='x', y='y+ci', color='r', linestyle='--', ax=axs[i])
        axs[i].grid()
        axs[i].set_title("f[{0}]".format(i))

      
def plot_predicted_vs_real(y_list: list, legends: list, mse:str):
    fig, axs = plt.subplots(1, len(y_list))
    fig.suptitle("MSE on prediction = {0}".format(mse), fontsize=16)
    for i, term in enumerate(y_list):
        axs[i].plot(y_list[i])
        axs[i].grid()
        axs[i].set_title(legends[i])
    
    
def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True).squeeze(), y_test.reset_index(drop=True).squeeze()


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
def generate_homoskedastic_normal_data(nrows=25000, output_folder = None):
    """
    Returns a pair of X,y to be used with NeuralGAM
    :param: nrows: data size (number of rows)
    :param: output_folder: folder path to save the generated files locally in CSV format
    :return: X: pandas Dataframe object with generated X (one column per feature). Xs follow a normal distribution
    :return: y: pandas Series object with computed y, with a normal distribution + homoskedastic residual
    """
    x1 = np.array(-10 + np.random.normal((nrows))*10)
    x2 = np.array(-10 + np.random.normal((nrows))*10)
    x3 = np.array(-10 + np.random.normal((nrows))*10)
    b = np.ones(nrows)* 2

    X = pd.DataFrame([x1,x2,x3,b]).transpose()
    X_func = pd.DataFrame([x1*x1, 2*x2, np.sin(x3), b]).transpose()
    plot_partial_dependencies(X, X_func, "Original f(x)")
    
    print("Residuals")
    err = np.random.normal(loc=0, scale=0.5, size=nrows)
    print(pd.DataFrame(err).describe())
    
    y = pd.Series(x1*x1 + 2*x2 + np.sin(x3) + b) + err
    print("y = f(x1) + f(x2) + f(x3) + b =  x1^2 + 2x2 + sin(x3) + b")

    if output_folder:
        save(X, y, output_folder)
    
    return X, y


def generate_homoskedastic_uniform_data(nrows=25000, output_folder = None):
    """
    Returns a pair of X,y to be used with NeuralGAM
    :param: nrows: data size (number of rows)
    :param: output_folder: folder path to save the generated files locally in CSV format
    :return: X: pandas Dataframe object with generated X (one column per feature). Xs follow a “continuous uniform” distribution
    :return: y: pandas Series object with computed y, with a normal distribution + homoskedastic error
    """
    x1 = np.array(-10 + np.random.random((nrows))*10)
    x2 = np.array(-10 + np.random.random((nrows))*10)
    x3 = np.array(-10 + np.random.random((nrows))*10)
    b = np.ones(nrows)* 2

    X = pd.DataFrame([x1,x2,x3,b]).transpose()
    X.columns = list(X.columns)
    X_func = pd.DataFrame([x1*x1, 2*x2, np.sin(x3), b]).transpose()
    plot_partial_dependencies(X, X_func, "Dataset f(x)")
    plt.show(block=False)
    
    print("Residuals")
    err = np.random.normal(loc=0, scale=0.5, size=nrows)
    print(pd.DataFrame(err).describe())
    
    y = pd.Series(x1*x1 + 2*x2 + np.sin(x3) + b) + err
    print("y = f(x1) + f(x2) + f(x3) + b =  x1^2 + 2x2 + sin(x3) + b")
    
    X_train, X_test, y_train, y_test = split(X, y)
    
    if output_folder:
        save(X_train, X_test, y_train, y_test, output_folder) 
    
    return X_train, X_test, y_train, y_test 

    
def compute_edf(a: pd.Series, b: pd.Series):
    return scipy.stats.ttest_rel(a, b)
        
