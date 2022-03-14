import itertools
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from scipy.stats import truncnorm

#todo delete in prod!!
np.random.seed(343142)

""" PLOTTING aux functions """
def plot_predicted_vs_real(dataframe_list: list, legends: list, title:str, output_path=None):
    fig, axs = plt.subplots(1, len(dataframe_list))
    fig.suptitle(title, fontsize=16)
    for i, term in enumerate(dataframe_list):
        axs[i].plot(dataframe_list[i])
        axs[i].grid()
        axs[i].set_title(legends[i])

    if output_path:
        plt.savefig(output_path, dpi = 100)
        fig = plt.gcf()
        plt.show(block=False)

def plot_confusion_matrix(cm, classes,
                        output_file,
                        title='Confusion matrix',
                        cmap=plt.cm.tab20c):
    """
    Plot confusion matrix
    :param cm: (numpy) confusion matrix values
    :param output_file: (string) path to output file
    :param title: (string) plot title
    :param cmap: (cmap) color map to use
    :return: None

    FROM https://deeplizard.com/learn/video/km7pxKy4UHU
    """
    fig, axs = plt.subplots(1, 1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print("Confusion Matrix: ")
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.4f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_file)
    plt.show(block=False)

def plot_y_histogram(dataframe_list: list, legends: list, title:str, output_path=None):
    fig, axs = plt.subplots(1, len(dataframe_list))
    fig.suptitle(title, fontsize=16)
    for i, term in enumerate(dataframe_list):
        #r_values = list(range(dataframe_list[i].shape[0]))
        axs[i].hist(dataframe_list[i])
        axs[i].grid()
        axs[i].set_title(legends[i])
    
    if output_path:
        plt.savefig(output_path, dpi = 100)
        fig = plt.gcf()
        plt.show(block=False)
    
def plot_multiple_partial_dependencies(x_list, f_list, legends, title, output_path=None):
    fig, axs = plt.subplots(nrows=1, ncols=len(f_list[0].columns))
    fig.suptitle(title, fontsize=16)
    for i, term in enumerate(f_list[0].columns):
        data = pd.DataFrame()
        for j in range(len(x_list)):
            data['x'] = x_list[j][x_list[j].columns[i]]
            data['y']= f_list[j][f_list[j].columns[i]]
            sns.lineplot(data = data, x='x', y='y', ax=axs[i])
        
        axs[i].legend(legends)
        axs[i].grid()
        axs[i].set_title("f[{0}]".format(i))
    
    if output_path:
        plt.savefig(output_path, dpi = 100)
        fig = plt.gcf()
        plt.show(block=False)


def plot_partial_dependencies(x, fs, title:str, output_path=None):
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
  
    if output_path:
        plt.savefig(output_path, dpi = 100)
        fig = plt.gcf()
        plt.show(block=False)


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

def generate_err(nrows:int, data_type:str, X:pd.DataFrame):
    if data_type == "homoscedastic":
        err = np.random.normal(loc=0, scale=0.2, size=nrows)
    elif data_type == "heteroscedastic":
        x_sum = X.sum(axis=1)
        err = np.random.normal(loc=0, scale=np.abs(0.01*x_sum), size=nrows)

    print("\n Intercept: {0} data".format(data_type))
    print(pd.DataFrame(err).describe())
    print(err)
    return err

def get_truncated_normal(mean=0, sd=1, low=0, upp=10, nrows=25000):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(nrows)
    
def generate_normal_data(nrows, data_type, link, output_path=""):
    
    x1 = get_truncated_normal(mean=0.0, sd=1.0, low=-5, upp=5, nrows=nrows)
    x2 = get_truncated_normal(mean=-0.5, sd=1.0, low=-5, upp=5, nrows=nrows)
    x3 = get_truncated_normal(mean=0.0, sd=1.0, low=-10, upp=5, nrows=nrows)

    X = pd.DataFrame([x1,x2,x3]).transpose()
    fs = pd.DataFrame([x1*x1, 2*x2, np.sin(x3)]).transpose()
    
    beta0 = generate_err(nrows=nrows, data_type=data_type, X=X)
    print("y = beta0 + f(x1) + f(x2) + f(x3) =  beta0 + x1^2 + 2x2 + sin(x3)")
    
    if link == "binomial":
        plot_partial_dependencies(X, fs, "Theoretical Model", output_path=output_path + "/thoeretical_model.png")
    
    y = compute_y(x1*x1 + 2*x2 + np.sin(x3), beta0, link)
    return X, y, fs

def generate_uniform_data(nrows, data_type, link, output_path = ""):
    
    x1 = np.array(np.random.uniform(low=-5, high=5, size=nrows))
    x2 = np.array(np.random.uniform(low=-10, high=5, size=nrows))
    x3 = np.array(np.random.uniform(low=-5, high=5, size=nrows))
    
    X = pd.DataFrame([x1,x2,x3]).transpose()
    fs = pd.DataFrame([x1*x1, 2*x2, np.sin(x3)]).transpose()
    
    if link == "binomial":
        plot_partial_dependencies(X, fs, "Theoretical Model", output_path=output_path + "/thoeretical_model.png")
        
    beta0 = generate_err(nrows=nrows, data_type=data_type, X=X)
    print("y = beta0 + f(x1) + f(x2) + f(x3) =  beta0 + x1^2 + 2x2 + sin(x3)")

    y = compute_y(x1*x1 + 2*x2 + np.sin(x3), beta0, link)
    return X, y, fs


def compute_y(x, beta0, link):
    y = pd.Series(x)
    y = y + beta0   
    if link == "binomial":
        # Apply a logit link function to transform y to binomial [0,1]
        y = pd.Series(np.exp(y)/(1+np.exp(y)))
    return y

def generate_data(type, distribution, link, nrows=25000, output_folder = ""):
    """
        Returns a pair of X,y to be used with NeuralGAM
        :param: type: homogeneity of variance on the intercept term {homoscedastic, heteroscedastic}
        :param: distribution: generate normal or uniform distributed X data {uniform, normal}
        :param: link: generate reponse Y in a continuous or binomial distribution
        :param: nrows: data size (number of rows)
        :param: output_folder: folder path to save the generated files locally in CSV format
        :return: X: pandas Dataframe object with generated X (one column per feature). Xs follow a normal distribution
        :return: y: pandas Series object with computed y, with a normal distribution + homoskedastic residual
    """
    
    if distribution == "uniform":
        X, y, fs = generate_uniform_data(nrows, data_type, link, output_path=output_folder)

    elif distribution == "normal":
        X, y, fs = generate_normal_data(nrows, data_type, link, output_path=output_folder)   

    return X, y, fs
    
def compute_edf(a: pd.Series, b: pd.Series):
    return scipy.stats.ttest_rel(a, b)
        