import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from src.utils.utils import generate_data, split, youden
from src.NeuralGAM.ngam import NeuralGAM
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser()


parser.add_argument(
    "-i",
    "--iteration",
    default=None,
    type=int,
    metavar="N_iteration (for simulations)"
)
parser.add_argument(
    "-o",
    "--output",
    default="results",
    dest="output",
    type=str,
    help="""Output folder"""
)

parser.add_argument(
    "-c",
    "--convergence_threshold",
    default=0.00001,
    type=float,
    metavar="Convergence Threshold of backfitting algorithm"
)
parser.add_argument(
    "-u",
    "--units",
    default=1024,
    type=str,
    dest="units",
    help="""Number of hidden units (i.e. 1024). If a list of values is provided, the neural network will add one hidden layer witch each number of units [1024, 512, 256] will generate a DNN with three hidden layers"""
)

parser.add_argument(
    "-l",
    "--lr",
    default=0.0053,
    type=float,
    dest="lr",
    help="""Learning Rate"""
)

if __name__ == "__main__":

    """
    LINEAR REGRESSION:
        python main.py linear -t {homoscedastic, heteroscedastic} -d {uniform, normal}
    LOGISTIC REGRESSION:
        python main.py logistic -d {uniform, normal}
    """

    args = parser.parse_args()
    variables = vars(args)

    print(variables)
    iteration = variables["iteration"]
    output_folder = variables.pop("output", "results")

    conv_threshold = variables.pop("convergence_threshold", 0.01)
    delta_threshold = variables.pop("delta_threshold", 0.00001)
    lr = variables.pop("lr", 0.0053)
    
    units = [int(item) for item in str(variables.pop("units")).split(',')]
    variables.pop("iteration")

    if iteration is not None:
        rel_path = "./{0}/{1}".format(output_folder, iteration)
        path = os.path.normpath(os.path.abspath(rel_path))
        #add iteration
        if not os.path.exists(path):
            os.makedirs(path)

    else:
        rel_path = "./{0}/".format(output_folder)
        path = os.path.normpath(os.path.abspath(rel_path))

    # add exec type
    data_type_path = "google"
    path = path + "/" + data_type_path
    if not os.path.exists(path):
        os.makedirs(path)

    """ Load dataset """

    print(f"Using {data_type_path}")
    
    X_train = pd.read_csv("./dataset/{0}/X_train.csv".format(data_type_path), index_col=0).reset_index(drop=True)
    fs_train = pd.read_csv("./dataset/{0}/fs_train.csv".format(data_type_path), index_col=0).reset_index(drop=True)
    y_train = pd.read_csv("./dataset/{0}/y_train.csv".format(data_type_path), index_col=0).reset_index(drop=True).squeeze()

    X_test = pd.read_csv("./dataset/{0}/X_test.csv".format(data_type_path), index_col=0).reset_index(drop=True)
    fs_test = pd.read_csv("./dataset/{0}/fs_test.csv".format(data_type_path), index_col=0).reset_index(drop=True)
    y_test = pd.read_csv("./dataset/{0}/y_test.csv".format(data_type_path), index_col=0).reset_index(drop=True).squeeze()

    ngam = NeuralGAM(num_inputs = len(X_train.columns), family="gaussian", num_units=units, learning_rate=lr)
    tstart = datetime.now()
    muhat, gs_train, eta = ngam.fit(X_train = X_train,
                            y_train = y_train,
                            max_iter_ls = 10,
                            bf_threshold=conv_threshold,
                            ls_threshold=delta_threshold,
                            max_iter_backfitting=10)
    tend = datetime.now()

    training_seconds = (tend - tstart).seconds
    variables["eta0"] = ngam.eta0
    variables["training_seconds"] = training_seconds
    variables["units"] = units
    variables["bf-threshold"] = conv_threshold
    variables["ls-threshold"] = delta_threshold
    variables["lr"] = lr
    
    y_pred, eta_pred = ngam.predict(X_test)
    fs_pred = ngam.get_partial_dependencies(X_test)

    err = mean_squared_error(y_train, ngam.eta)
    pred_err = mean_squared_error(y_test, eta_pred)

    variables["err"] = err
    variables["err_test"] = pred_err
    variables["err_rmse"] = np.sqrt(err)
    variables["err_test_rmse"] = np.sqrt(pred_err)
    print("Done")
    print(variables)

    """ SAVE RESULTS"""
    pd.DataFrame(fs_pred).to_csv(path + "/fs_test_estimated.csv")
    pd.DataFrame(y_pred).to_csv(path + "/y_pred.csv")

    pd.DataFrame(gs_train).to_csv(path + "/fs_train_estimated.csv")
    pd.DataFrame.from_dict(variables, orient="index").transpose().to_csv(path + "/variables.csv", index=False)