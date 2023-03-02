
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
plt.style.use('seaborn')

params = {"axes.linewidth": 2,
        "font.family": "serif"}

matplotlib.rcParams['agg.path.chunksize'] = 10000
plt.rcParams.update(params)


np.random.seed(343142)

def plot_predicted_vs_real(dataframe_list: list, legends: list, title:str, output_path=None):
    fig, axs = plt.subplots(1, len(dataframe_list))
    fig.suptitle(title, fontsize=10)
    for i, term in enumerate(dataframe_list):
        axs[i].plot(dataframe_list[i])
        axs[i].grid()
        axs[i].set_title(legends[i])

    if output_path:
        plt.savefig(output_path, dpi = 300, bbox_inches = "tight")
        fig = plt.gcf()

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
    plt.savefig(output_file, dpi=300, bbox_inches = "tight")
    plt.show(block=False)

def plot_y_histogram(dataframe_list: list, legends: list, title:str, output_path=None):
    fig, axs = plt.subplots(1, len(dataframe_list))
    fig.suptitle(title, fontsize=10)
    for i, term in enumerate(dataframe_list):
        #r_values = list(range(dataframe_list[i].shape[0]))
        axs[i].hist(dataframe_list[i])
        axs[i].grid()
        axs[i].set_title(legends[i])
    
    if output_path:
        plt.savefig(output_path, dpi = 300, bbox_inches = "tight")
        fig = plt.gcf()

 
def experiments_plot_partial_dependencies(x_list, f_list, legends, title, output_path=None):    
    fig, axs = plt.subplots(nrows=1, ncols=len(f_list[0].columns), figsize=(25,20))
    fig.suptitle(title)

    for i, term in enumerate(f_list[0].columns):
        data = pd.DataFrame()
        for j in range(len(x_list)):
            if j==0:
                color = "royalblue"
                style = "-"
            else:
                color = "mediumseagreen"
                style = "--"

            data['x'] = x_list[j][x_list[j].columns[i]]
            data['y']= f_list[j][f_list[j].columns[i]]
            sns.lineplot(data = data, x='x', y='y', ax=axs[i], color=color, linestyle=style)
            
        axs[i].grid()
        axs[i].set_xlabel(f"$X_{i+1}$", fontsize=30)
        axs[i].set_ylabel(f"$f(x_{i+1})$", fontsize=30)
    
    import matplotlib.patches as mpatches

    theoretical_patch = mpatches.Patch(color='red', label='Theoretical f(x)')
    learned_patch = mpatches.Patch(color='green', label='Learned f(x) from Neural GAM')
    
    plt.tight_layout()
    fig.legend(handles=[theoretical_patch, learned_patch], loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    
    if output_path:
        plt.savefig(output_path, bbox_inches = "tight")
        fig = plt.gcf()


def youden(y_true, y_score):
    """
    Find data-driven cut-off for classification
    
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    
    Parameters
    ----------
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
        
    References
    ----------
    
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    
    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]
    
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

def generate_err(nrows:int, data_type:str, eta0:pd.DataFrame):
    err = np.random.normal(loc=0, scale=0.5, size=nrows)
    if data_type == "heteroscedastic":
        #sigma = np.sqrt(0.5 + 0.05 * np.abs(eta0))
        sigma = 0.5 + np.abs(0.25*eta0)
        err = err * sigma

    print("\n Intercept: {0} data".format(data_type))
    print(pd.DataFrame(err).describe())

    return err

def get_truncated_normal(mean=0, sd=1, low=0, upp=10, nrows=25000):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(nrows)
    
def generate_normal_data(nrows, data_type, family, output_path=""):
    
    x1 = get_truncated_normal(mean=0.0, sd=1.0, low=-5, upp=5, nrows=nrows)
    x2 = get_truncated_normal(mean=0.0, sd=1.0, low=-5, upp=5, nrows=nrows)
    x3 = get_truncated_normal(mean=0.0, sd=1.0, low=-5, upp=5, nrows=nrows)
    beta0 = np.ones(nrows) * 2
    
    X = pd.DataFrame([x1,x2,x3]).transpose()
    fs = pd.DataFrame([x1*x1, 2*x2, np.sin(x3)]).transpose()
    print("y = beta0 + f(x1) + f(x2) + f(x3) =  2 + x1^2 + 2x2 + sin(x3)")
   
    y = compute_y(fs, beta0, nrows, data_type, family)
    
    return X, y, fs

def generate_uniform_data(nrows, data_type, family, output_path = ""):
    
    x1 = np.array(np.random.uniform(low=-2.5, high=2.5, size=nrows))
    x2 = np.array(np.random.uniform(low=-2.5, high=2.5, size=nrows))
    x3 = np.array(np.random.uniform(low=-2.5, high=2.5, size=nrows))
    beta0 = np.ones(nrows) * 2 
    X = pd.DataFrame([x1,x2,x3]).transpose()
    fs = pd.DataFrame([x1*x1, 2*x2, np.sin(x3)]).transpose()
    print("y = beta0 + f(x1) + f(x2) + f(x3) =  2 + x1^2 + 2x2 + sin(x3)")   
    y = compute_y(fs, beta0, nrows, data_type, family)
    
    return X, y, fs


def compute_y(fs, beta0, nrows, data_type, family):
    
    y = fs.sum(axis=1) + beta0
    
    if family == "binomial":
        y = y - np.mean(y)
        y = np.exp(y)/(1+np.exp(y)) # Probabilities of success       
        
    elif family == "gaussian":
        err = generate_err(nrows=nrows, data_type=data_type, eta0=y)
        y = y + err
        y = y - np.mean(y)
    return pd.Series(y)

def generate_data(type, distribution, family, nrows=25000, output_folder = ""):
    """
        Returns a pair of X,y to be used with NeuralGAM
        :param: type: homogeneity of variance on the intercept term {homoscedastic, heteroscedastic}
        :param: distribution: generate normal or uniform distributed X data {uniform, normal}
        :param: family: generate reponse Y for linear or binomial regression problems
        :param: nrows: data size (number of rows)
        :param: output_folder: folder path to save the generated files locally in CSV format
        :return: X: pandas Dataframe object with generated X (one column per feature). Xs follow a normal distribution
        :return: y: pandas Series object with computed y, with a normal distribution + homoskedastic residual
    """
    
    if distribution == "uniform":
        X, y, fs = generate_uniform_data(nrows, type, family, output_path=output_folder)

    elif distribution == "normal":
        X, y, fs = generate_normal_data(nrows, type, family, output_path=output_folder)   

    return X, y, fs
    
def compute_edf(a: pd.Series, b: pd.Series):
    return scipy.stats.ttest_rel(a, b)
        