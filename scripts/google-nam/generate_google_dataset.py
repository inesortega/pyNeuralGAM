"""
This script generates a dataset to simulate the experiment from NAM paper:

[1] R. Agarwal et al., “Neural Additive Models: Interpretable Machine Learning with Neural Nets,” 
in Advances in Neural Information Processing Systems, 2021, vol. 34, pp. 4699–4711, [Online]. 
Available: https://proceedings.neurips.cc/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf.
"""

import numpy as np
import pandas as pd

nrows = 25000

# generate Xs 

x1 = np.array(np.random.uniform(low=-1, high=1, size=nrows))
x2 = np.array(np.random.uniform(low=-1, high=1, size=nrows))
x3 = np.array(np.random.uniform(low=-1, high=1, size=nrows))

X = pd.DataFrame([x1,x2,x3]).transpose()

f1 = 1/3 * (np.log(100*x1 + 101))
f2 = -4/3 * np.exp(-4*np.abs(x2))
f3 = np.sin(10*x3)

fs = pd.DataFrame([f1, f2, f3]).transpose()

err = np.random.normal(loc=0, scale=5/6, size=nrows)

y = fs.sum(axis=1)
y = y + err
y = y - np.mean(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
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