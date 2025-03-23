import numpy as np
from sklearn.metrics import roc_curve
import scipy
import pandas as pd

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

def compute_edf(a: pd.Series, b: pd.Series):
    """
    Compute the empirical degrees of freedom (EDF) between two pandas Series.
    This function calculates the t-statistic and p-value for the differences
    between two pandas Series `a` and `b`. It assumes that the differences
    follow a normal distribution.
    Parameters:
        a (pd.Series): The first pandas Series.
        b (pd.Series): The second pandas Series.
    Returns:
        tuple: A tuple containing the t-statistic and the p-value.
    """
    
    # Calculate the differences
    differences = a - b
    
    # Calculate the mean of the differences
    mean_diff = np.mean(differences)
    
    # Calculate the standard deviation of the differences
    std_diff = np.std(differences, ddof=1)
    
    # Calculate the number of differences
    n = len(differences)
    
    # Calculate the t-statistic
    t_statistic = mean_diff / (std_diff / np.sqrt(n))
    
    # Calculate the degrees of freedom
    df = n - 1
    
    # Calculate the p-value
    p_value = 2 * (1 - scipy.stats.t.cdf(np.abs(t_statistic), df))
    
    return t_statistic, p_value
        