import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import l
from sklearn.model_selection import train_test_split
from src.NeuralGAM.ngam import NeuralGAM, load_model
import argparse
import seaborn as sns

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='subparser')

parser_run = subparsers.add_parser('train', help='train NeuralGAM model')
parser_run = subparsers.add_parser('test', help='test NeuralGAM model')
parser_run = subparsers.add_parser('generate_data', help='generate data to test/train NeuralGAM model')

def train():
    
    X = pd.read_csv("./test/data/x_train.csv")
    y = pd.read_csv("./test/data/y_train.csv", squeeze=True)

    ngam = NeuralGAM(num_inputs = len(X.columns), num_units=64)

    ycal, mse = ngam.fit(X_train = X, y_train = y, max_iter = 50)

    print("Beta0 {0}".format(ngam.beta))
    
    ngam.save_model("./output.ngam")
   
    
def test():
    
    ngam = load_model("./output.ngam")
    
    X_test = pd.read_csv("./test/data/x_test.csv")
    y_test = pd.read_csv("./test/data/y_test.csv", squeeze=True)
    
    fs = ngam.compute_X(X_test)
    
    y_pred = ngam.predict(X_test)
    
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    
    plot_predicted_vs_real([y_test, y_pred], ["real", "predicted"], mse=mse)
    plot_partial_dependencies(X_test, fs)
    plt.show(block=True)
    
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
       
def plot_predicted_vs_real(y_list: list, legends: list, mse:str):
    fig, axs = plt.subplots(1, len(y_list))
    fig.suptitle("MSE on prediction = {0}".format(mse), fontsize=16)
    for i, term in enumerate(y_list):
        axs[i].plot(y_list[i])
        axs[i].grid()
        axs[i].set_title(legends[i])
    
def plot_partial_dependencies(x, fs):
    fig, axs = plt.subplots(nrows=1, ncols=len(fs.columns))
    fig.suptitle("Learned functions with confidence intervals at 95%", fontsize=16)
    for i, term in enumerate(fs.columns):
        x_i = x[x.columns[i]]
        y = fs[fs.columns[i]]
        
        # calculate confidence interval at 95%
        ci = 1.96 * np.std(y)/np.sqrt(len(x_i))
        
        sns.lineplot(x_i, y, color='b', ax=axs[i])
        sns.lineplot(x_i, (y-ci), color='r', linestyle='--', ax=axs[i])
        sns.lineplot(x_i, (y+ci), color='r', linestyle='--', ax=axs[i])
        axs[i].set_title("f[{0}]".format(i))

if __name__ == "__main__":
    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop('subparser')](**kwargs)