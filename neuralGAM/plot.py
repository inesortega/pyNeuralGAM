# Description: This file contains functions to plot the results of the neuralGAM model.
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
import numpy as np 

def plot_partial_dependencies(x: pd.DataFrame, fs: pd.DataFrame, title: str = None, output_path: str = None, label = None):    
    """
    Plots partial dependency plots for each feature in the dataset.
    Parameters:
        x (pd.DataFrame): DataFrame containing the feature values.
        fs (pd.DataFrame): DataFrame containing the partial dependency values for each feature.
        title (str): Title of the plot.
        output_path (str, optional): Path to save the plot. If None, the plot is displayed. Defaults to None.
    Returns:
        None
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(fs.columns), figsize=(10,8))
    fig.suptitle(title)
    
    for i, term in enumerate(fs.columns):
        data = pd.DataFrame()
        data['x'] = x[x.columns[i]]
        data['f(x)']= fs[fs.columns[i]]
        
        sns.lineplot(data = data, x='x', y='f(x)', ax=axs[i], legend=label)
        axs[i].grid()
        axs[i].set_xlabel(f'$X_{i+1}$')
        axs[i].set_ylabel(f'$f(x_{i+1})$')
        if label:
            axs[i].legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi = 300, bbox_inches = "tight")
        fig = plt.gcf()
    else:
        plt.show()
    plt.close
    return

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
    return
def experiments_plot_partial_dependencies(x_list, f_list, legends, title, output_path=None):    
    """
    Plots partial dependencies for a list of experiments. This method is useful for simmulated data when we have the X and f(x) values for different experiments.
    Parameters:
        x_list (list of pd.DataFrame): List of DataFrames containing the input features for each experiment.
        f_list (list of pd.DataFrame): List of DataFrames containing the partial dependency values for each experiment.
        legends (list of str): List of legend labels for the experiments.
        title (str): Title of the plot.
        output_path (str, optional): Path to save the plot. If None, the plot is not saved.
    Returns:
    None
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(f_list[0].columns), figsize=(10,8))
    fig.suptitle(title)

    for i, term in enumerate(f_list[0].columns):
        for j in range(len(x_list)):
            data = pd.DataFrame()
            data['x'] = x_list[j][x_list[j].columns[i]]
            data['y'] = f_list[j][f_list[j].columns[i]]
            sns.lineplot(data=data, x='x', y='y', ax=axs[i], label=legends[j])
            
        axs[i].grid()
        axs[i].set_xlabel(f"$X_{i+1}$")
        axs[i].set_ylabel(f"$f(x_{i+1})$")
        axs[i].legend()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        fig = plt.gcf()
