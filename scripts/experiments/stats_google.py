from inspect import trace
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use('Agg')
import os
import seaborn as sns


if __name__ == "__main__":
    # get random number of iterations on range [1, 1000]

    rel_path = "./"
    path = os.path.normpath(os.path.abspath(rel_path))

    RANGE_SIZE = 1000
    print("generating statistics from {0} iterations...".format(RANGE_SIZE))

    plot_types = ["uniform_binomial", "heteroscedastic_uniform_gaussian", "homoscedastic_uniform_gaussian"]

    all_pred = np.zeros(RANGE_SIZE, dtype=object)   # for 0-1 loss you might need int64

    for type in plot_types:
        try:
            if not os.path.exists(type):
                os.mkdir(type)
            err_train = pd.DataFrame()
            err_pred = pd.DataFrame()
            training_seconds = pd.DataFrame()
            all_pred = pd.DataFrame()

            for j in range(1, RANGE_SIZE+1):
                variables = pd.read_csv("./results-nam/{0}/{1}/variables.csv".format(j, type), index_col=0).reset_index(drop=True)
                variables_training = pd.read_csv("./results-nam/{0}/{1}/variables_training.csv".format(j, type), index_col=0).reset_index(drop=True)
                err_train = pd.concat([err_train, variables["err"]], axis=0)
                err_pred = pd.concat([err_pred, variables["err_test"]], axis=0)
                err_train_rmse = pd.concat([err_train, variables["err_rmse"]], axis=0)
                err_pred_rmse = pd.concat([err_pred, variables["err_test_rmse"]], axis=0)
                training_seconds = pd.concat([training_seconds, variables_training["training_seconds"]], axis=0)
                y_pred = pd.read_csv("./results-nam/{0}/{1}/y_pred.csv".format(j, type), index_col=0).reset_index(drop=True)
                all_pred = pd.concat([all_pred, y_pred], axis=1)

            err_train.to_csv("./{0}/err_train.csv".format(type), index=False)
            err_pred.to_csv("./{0}/err_pred.csv".format(type), index=False)
            err_train_rmse.to_csv("./{0}/err_train_rmse.csv".format(type), index=False)
            err_pred_rmse.to_csv("./{0}/err_pred_rmse.csv".format(type), index=False)
            training_seconds.to_csv("./{0}/training_seconds.csv".format(type), index=False)

            # obtain pred to compute mean bias / variance
            y_test = pd.read_csv("./dataset/{0}/y_test.csv".format(type), index_col=0).reset_index(drop=True)["0"].values

            """https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/bias_variance_decomp.py"""
            """
            loss = np.apply_along_axis(lambda x: ((x - y_test) ** 2).mean(), axis=0, arr=all_pred)
            avg_expected_loss = loss.mean()
            mean_predictions = np.mean(all_pred, axis=1)
            avg_bias = np.sum((mean_predictions - y_test) ** 2) / y_test.size
            avg_var = avg_expected_loss - avg_bias**2
            """

            # Bias: for each x, mean of 1000 estimations of x_i - y_test (averaged)
            bias = (all_pred.mean(axis=1) - y_test).mean()
            # Variane: variance of 1000 estimations, for each x_i (averaged)
            var = all_pred.var(axis=1).mean()


            mean_values = {}
            mean_values["err_train"]  = err_train.mean()[0]
            mean_values["err_pred"]  = err_pred.mean()[0]
            mean_values["err_pred_rmse"]  = err_pred_rmse.mean()[0]
            mean_values["err_train_rmse"]  = err_train_rmse.mean()[0]
            mean_values["training_seconds"]  = training_seconds.mean()[0]
            mean_values["bias"] = bias
            mean_values["var"] = var
            
            pd.DataFrame.from_dict(mean_values, orient="index").transpose().to_csv("./{0}/mean_metrics_google.csv".format(type), index=False)

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            print("error on it type={0}".format(type))
            print(range)
