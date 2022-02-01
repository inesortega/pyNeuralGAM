# import libraries
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.NN.convolutional import build_multivariate_model, build_feature_model
from src.GAM.gam import fit_gam


def plot_partial_dependencies(x, y_pred):
    
    fig, axs = plt.subplots(nrows=1, ncols=len(gam.terms)-1)
    fig.suptitle("Partial dependency plots with confidence intervals at 95%", fontsize=16)
    for i, term in enumerate(x.columns):
        
        #some confidence interval
        ci = 0.95 * np.std(y_pred)/np.sqrt(len(x[i]))
        axs[i].plot(x[i], y_pred)
        axs[i].plot(x[i], ci, c='r', ls='--')
        axs[i].grid()
        axs[i].set_title(term)

# Generate training data (random values between -50 and 50 )
x1 = np.array(-50 + np.random.random((25000))*50)
x2 = np.array(-50 + np.random.random((25000))*50)
x3 = np.array(-50 + np.random.random((25000))*50)

X = pd.DataFrame([x1,x2,x3]).transpose()

# y = f(x1) + f(x2) + f(x3) =  x1^2 + 2x2 + sin(x3).
y = pd.Series(x1*x1 + 2*x2 + np.sin(x3))

# use data split and fit to run the model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

ycal, models, alpha = fit_gam(x_train, y_train)

print("Alpha {0}".format(alpha))

for i,feature in enumerate(x_test.columns):
    y_pred = models[i].predict(x_test[i])
    plot_partial_dependencies

