import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.NeuralGAM.ngam import NeuralGAM, load_model
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='subparser')

parser_run = subparsers.add_parser('train', help='train NeuralGAM model')
parser_run = subparsers.add_parser('test', help='test NeuralGAM model')
parser_run = subparsers.add_parser('generate_data', help='generate data to test/train NeuralGAM model')

def train():
    
    X = pd.read_csv("./test/data/x_train.csv")
    y = pd.read_csv("./test/data/y_train.csv", squeeze=True)

    ngam = NeuralGAM(num_inputs = len(X.columns), num_units=64)

    ycal, mse = ngam.fit(X, y, 1, None, 100)

    print("Beta0 {0}".format(ngam.beta))
    
    ngam.save_model("./output.gam")
   
    
def tests():
    
    ngam = NeuralGAM(load_model("./output.gam"))
    
    X_test = pd.read_csv("./test/data/x_test.csv")
    y_test = pd.read_csv("./test/data/y_test.csv", squeeze=True)
    
    y_pred = ngam.predict(X_test)
    
def generate_data():
    
    x1 = np.array(-10 + np.random.random((25000))*10)
    x2 = np.array(-10 + np.random.random((25000))*10)
    x3 = np.array(-10 + np.random.random((25000))*10)
    b = np.array(-10 + np.random.random((25000))*10)

    X = pd.DataFrame([x1,x2,x3, b]).transpose()

    # y = f(x1) + f(x2) + f(x3) =  x1^2 + 2x2 + sin(x3).
    y = pd.Series(x1*x1 + 2*x2 + np.sin(x3) + b)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    X_train.to_csv("./test/data/x_train.csv", index=False)
    X_test.to_csv("./test/data/x_test.csv", index=False)
    y_train.to_csv("./test/data/y_train.csv", index=False)
    y_test.to_csv("./test/data/y_test.csv", index=False)
       
def plot_predicted_vs_real(y_real, y_pred):
    fig = plt.plot()
    fig.plot(y_real, c='r', ls='--', legend="Real y")
    fig.plot(y_pred, c='b', ls='.', legend="Predicted y")
    fig.grid()
    fig.set_title("Predicted vs Real function plot")
    
def plot_partial_dependencies(x, y_pred):
    
    fig, axs = plt.subplots(nrows=1, ncols=len(x.columns)-1)
    fig.suptitle("Partial dependency plots with confidence intervals at 95%", fontsize=16)
    for i, term in enumerate(x.columns):
        
        #some confidence interval
        ci = 0.95 * np.std(y_pred)/np.sqrt(len(x[i]))
        axs[i].plot(x[i], y_pred)
        axs[i].plot(x[i], ci, c='r', ls='--')
        axs[i].grid()
        axs[i].set_title(term)
        
        
if __name__ == "__main__":
    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop('subparser')](**kwargs)