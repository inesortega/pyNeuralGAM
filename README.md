## neuralGAM: Interpretable Neural Network Based on Generalized Additive Models

Neural Networks are one of the most popular methods nowadays given their high performance on diverse tasks, such as computer vision, anomaly detection, computer-aided disease detection and diagnosis, or natural language processing. However, it is usually unclear how neural networks make decisions, and current methods that try to provide interpretability to neural networks are not robust enough.

We introduce **neuralGAM**, a fully explainable neural network based on **Generalized Additive Models**, which trains a different neural network to estimate the contribution of each feature to the response variable. The networks are trained independently leveraging the local scoring and backfitting algorithms to ensure that the Generalized Additive Model converges and it is additive. The resultant model is a highly accurate and explainable deep learning model, which can be used for high-risk AI practices where decision-making should be based on accountable and interpretable algorithms.

**neuralGAM** is also available as an R package at the [CRAN](https://cran.r-project.org/package=neuralGAM)

## Installation

To install the neuralGAM package, you can use the following command:

```sh
pip install neuralGAM
```

## Usage

### Linear Regression

To perform linear regression using the neuralGAM package, follow these steps:

1. Import the necessary libraries and the NeuralGAM class:

    ```python
    from neuralGAM.model import NeuralGAM
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    ```

2. Load your dataset and split it into training and testing sets:

    ```python
    data = pd.read_csv('path/to/your/dataset.csv')
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

3. Initialize the NeuralGAM model. You might need to adjust the `num_units` parameter depending on your data complexity and availability. Each number in the list defines the number of hidden units in each layer of the Deep Neural Network. For simple problems we recommend a single-layer neural network with 1024 units.

    ```python
    ngam = NeuralGAM(family="gaussian", num_units=[1024], learning_rate=0.00053)
    ```

4. Fit the model to the training data:

    ```python
    muhat, fs_train_estimated, eta = ngam.fit(X_train=X_train, y_train=y_train, max_iter_ls=10, bf_threshold=1e-5, ls_threshold=0.01, max_iter_backfitting=10, parallel=True)
    ```

5. Make predictions on the test data and compute the mean squared error:

    ```python
    y_pred = ngam.predict(X_test, type="response")
    pred_err = mean_squared_error(y_test, y_pred)
    print(f"MSE in the test set = {pred_err}")
    ```

6. Plot the partial dependencies:

    ```python
    from neuralGAM.plot import plot_partial_dependencies
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8')
    plot_partial_dependencies(x=X_train, fs=fs_train_estimated, title="Estimated Training Partial Effects")
    fs_test_est = ngam.predict(X_test, type="terms")
    plot_partial_dependencies(x=X_test, fs=fs_test_est, title="Estimated Test Partial Effects")
    ```

### Logistic Regression

To perform logistic regression using the neuralGAM package, follow these steps:

1. Import the necessary libraries and the NeuralGAM class:

    ```python
    from neuralGAM.model import NeuralGAM
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    ```

2. Load your dataset and split it into training and testing sets:

    ```python
    data = pd.read_csv('path/to/your/dataset.csv')
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

3. Initialize the NeuralGAM model. You might need to adjust the `num_units` parameter depending on your data complexity and availability. Each number in the list defines the number of hidden units in each layer of the Deep Neural Network. For simple problems we recommend a single-layer neural network with 1024 units.

    ```python
    ngam = NeuralGAM(family="binomial", num_units=[1024], learning_rate=0.00053)
    ```

4. Fit the model to the training data:

    ```python
    muhat, fs_train_estimated, eta = ngam.fit(X_train=X_train, y_train=y_train, max_iter_ls=10, bf_threshold=1e-5, ls_threshold=0.01, max_iter_backfitting=10, parallel=True)
    ```

5. Make predictions on the test data and compute the accuracy:

    ```python
    y_pred = ngam.predict(X_test, type="response")
    y_pred_class = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Accuracy in the test set = {accuracy}")
    ```

6. Plot the partial dependencies:

    ```python
    from neuralGAM.plot import plot_partial_dependencies
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8')
    plot_partial_dependencies(x=X_train, fs=fs_train_estimated, title="Estimated Training Partial Effects")
    fs_test_est = ngam.predict(X_test, type="terms")
    plot_partial_dependencies(x=X_test, fs=fs_test_est, title="Estimated Test Partial Effects")
    ```

## Examples

You can find detailed examples for both linear and logistic regression in the examples folder. These examples are provided as Jupyter notebooks:

- Linear Regression Example
- Logistic Regression Example

## Citation

If you use neuralGAM in your research, please cite the following paper:

> Ortega-Fernandez, I., Sestelo, M. & Villanueva, N.M. Explainable generalized additive neural networks with independent neural network training. Stat Comput 34, 6 (2024). https://doi.org/10.1007/s11222-023-10320-5

```bibtex
@article{ortega2024explainable,
  title={Explainable generalized additive neural networks with independent neural network training},
  author={Ortega-Fernandez, Ines and Sestelo, Marta and Villanueva, Nora M},
  journal={Statistics and Computing},
  volume={34},
  number={1},
  pages={6},
  year={2024},
  publisher={Springer}
}
```