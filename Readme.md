# Neural GAM: Neural Generalized Additive Models

  **[Overview](#overview)**
| **[NeuralGAM Visualization](#neuralgam-visualization)**
| **[Usage](#usage)**


NeuralGAM is a project for Generalized Additive Models (GAM) research. We provide a library which implements Neural GAMs: a way of fitting a Generalized Additive Model by learning a linear combination of Deep Neural Networks. GAMs are a class of non-parametric regression models, where each input feature is a smooth function. 

Each neural network attends to a single input feature. The GAM is fitted using the backfitting algorithm, where a Deep Neural Network is fitted to learn a smooth function representing the smottthed fit for the residuals of all the others variables. 

## Overview

```python
from src.NeuralGAM.ngam import NeuralGAM
from src.utils.utils import generate_data, plot_multiple_partial_dependencies
```

##### Generate sample data

We provide scripts to generate datasets for testing NeuralGAM for regression and classification problems. 

| Parameter     | Description                                    | Values |
| -----------   | ---------------------------------------------- | ------------------------- |
| type          | homogeneity of variance on the intercept term  | {homoscedastic, heteroscedastic}       |
| distibution   | distribution of the X                          | {normal, uniform} 
| link          | Link function to apply (binomial for classification problems, identity for regression) | {identity, link}                   |


```python

X, y, fs = generate_data(nrows=25000, type=type, distribution=distribution, link=link, output_folder=path)
X_train, X_test, y_train, y_test = split(X, y)

ngam = NeuralGAM(num_inputs = len(X_train.columns), link=link)
ycal, mse = ngam.fit(X_train = X_train, y_train = y_train, max_iter = 5, convergence_threshold=0.04)
ngam.save_model(path)

y_pred = ngam.predict(X_test)

# For classification problems, transform into 0/1:

y_test = np.where(y_test >= 0.5, 1, 0)
y_pred = np.where(y_pred >= 0.5, 1, 0) 

```

## NeuralGAM Visualization

The function `get_partial_dependencies´ provides the learned functions of the NeuralGAM model  after fitting: 
```python
test_fs = ngam.get_partial_dependencies(X_test)
training_fs = ngam.get_partial_dependencies(X_train)

x_list = [X_train, X_test]
fs_list = [training_fs, test_fs]
legends = ["X_train", "X_test"]
plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title="MSE = {0}".format(mse), output_path=path + "/partial_dep.png")
```

## Usage

```bash
$ python main.py -h
usage: main.py [-h] [-t {homoscedastic, heteroscedastic}] [-d {uniform, normal}] [-l {identity, binomial}]

optional arguments:
  -h, --help            show this help message and exit
  -t {homoscedastic, heteroscedastic} , --type {homoscedastic, heteroscedastic} 
                        Choose wether to generate a homoscesdastic or heteroscedastic dataset
  -d {uniform, normal} , --distribution {uniform, normal} 
                        Choose wether to generate normal or uniform distributed dataset
  -l {identity, binomial} , --link {identity, binomial} 
                        Choose wether the response Y is continuous (for regression problems) or binomial (for classification)
```
