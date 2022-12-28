import numpy as np
import pandas as pd

nrows = 29500

# generate Xs 

x1 = np.array(np.random.uniform(low=-1, high=1, size=nrows))
x2 = np.array(np.random.uniform(low=-1, high=1, size=nrows))
x3 = np.array(np.random.uniform(low=-1, high=1, size=nrows))
beta0 = np.ones(nrows) * 2

X = pd.DataFrame([x1,x2,x3]).transpose()

f1 = 1/3 * (np.log(100*x1 + 101))
f2 = -4/3 * np.exp(-4*np.abs(x2))
f3 = np.sin(10*x3)
#f4 = np.cos(15*x3)

fs = pd.DataFrame([f1, f2, f3]).transpose()

err = np.random.normal(loc=0, scale=5/6, size=nrows)

y = fs.sum(axis=1) + beta0
y = y + err
y = y - np.mean(y)

from sklearn.model_selection import train_test_split

""" 
12500 --- 100%
10000 --- x

x = 80% 
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5000, shuffle=True)
fs_train = fs.iloc[X_train.index].reset_index(drop=True)
fs_test = fs.iloc[X_test.index].reset_index(drop=True)

X_train =  X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True).squeeze()
y_test = y_test.reset_index(drop=True).squeeze()

data_type_path = "google"

pd.DataFrame(X_train).to_csv("./dataset/{0}/X_train.csv".format(data_type_path))
pd.DataFrame(y_train).to_csv("./dataset/{0}/y_train.csv".format(data_type_path))
pd.DataFrame(fs_train).to_csv("./dataset/{0}/fs_train.csv".format(data_type_path))
pd.DataFrame(X_test).to_csv("./dataset/{0}/X_test.csv".format(data_type_path))
pd.DataFrame(y_test).to_csv("./dataset/{0}/y_test.csv".format(data_type_path))
pd.DataFrame(fs_test).to_csv("./dataset/{0}/fs_test.csv".format(data_type_path))