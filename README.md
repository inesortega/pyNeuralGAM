# Neural GAM: Neural Generalized Additive Models

  **[Overview](#overview)**
| **[NeuralGAM Visualization](#neuralgam-visualization)**
| **[Usage](#usage)**


NeuralGAM is a project for Generalized Additive Models (GAM) research. We provide a library which implements Neural GAMs: a way of fitting a Generalized Additive Model by learning a linear combination of Deep Neural Networks. GAMs are a class of non-parametric regression models, where each input feature is a smooth function. 

![formula](https://latex.codecogs.com/svg.image?y=&space;beta_0&space;&plus;&space;\sum_{i=1}^{N}&space;f_i(x))

Each neural network attends to a single input feature. The NeuralGAM is fitted using the backfitting algorithm, where a Deep Neural Network is fitted one epoch at a time to learn a smooth function representing the smottthed fit for the residuals of all the others variables. 

## Overview

```python
from src.NeuralGAM.ngam import NeuralGAM
from src.utils.utils import generate_data, plot_multiple_partial_dependencies
```

##### Generate sample data

We provide scripts to generate simulation data to test NeuralGAM for linear/binomial regression problems. 

| Parameter     | Description                                    | Values |
| -----------   | ---------------------------------------------- | ------------------------- |
| type          | homogeneity of variance on the intercept term  | {homoscedastic, heteroscedastic}       |
| distibution   | distribution of the X                          | {normal, uniform} 
| link          | Link function to apply (binomial regression or identity for linear regression) | {identity, binomial}                   |

The fitting parameters are the maximum number of iterations to wait for function convergence (defaults to 5), and the convergence threshold (defaults to 0.04): when this value is reached on the Mean Squared Error between the target variable (y) and the current linear combination of learned features, the backfitting algorithm is stoped to avoid overfitting.

The simulation generates three different features, each modeling a function plus an intercept term, being the final theoretical model: 

![formula-2](https://latex.codecogs.com/svg.image?\inline&space;y&space;=&space;beta_0&space;&plus;&space;x_1^2&space;&plus;&space;2*x_2&space;&plus;&space;\sin(x_3)&space;)

```python

X, y, fs = generate_data(nrows=25000, type=type, distribution=distribution, link=link, output_folder=path)
X_train, X_test, y_train, y_test = split(X, y)

ngam = NeuralGAM(num_inputs = len(X_train.columns), link=link)
ycal, mse = ngam.fit(X_train = X_train, y_train = y_train, max_iter = 5, convergence_threshold=0.04)
ngam.save_model(path)

y_pred = ngam.predict(X_test)

# For classification problems, transform into classes (0 or 1) using numpy:

y_test = np.where(y_test >= 0.5, 1, 0)
y_pred = np.where(y_pred >= 0.5, 1, 0) 

```

## NeuralGAM Visualization

The function `get_partial_dependencies´ provides the learned functions of the NeuralGAM model  after fitting: 

```python
test_fs = ngam.get_partial_dependencies(X_test)
training_fs = ngam.get_partial_dependencies(X_train)

x_list = [X, X_train, X_test]
fs_list = [fs, training_fs, test_fs]
legends = ["theoretical_f", "X_train", "X_test"]
plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title="MSE = {0}".format(mse), output_path=path + "/partial_dep.png")
```

The following image shows the resultant theoretical model from the dataset (in blue), the learned functions for each feature after the training process (orange), and the predicted functions from the test set (green), for a linear regression simulation with heteroscedastic intercept and normally distributed data. 

![](functions.png)
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
