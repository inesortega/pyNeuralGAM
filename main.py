import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from src.utils.utils import generate_data, split
from src.NeuralGAM.ngam import NeuralGAM
import pandas as pd
from datetime import datetime
from src.utils.utils import plot_multiple_partial_dependencies

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help='Choose wether to use Linear (linear) or Logistic (logistic) Regression')

linear_regression_parser = subparsers.add_parser(name='linear', help="Linear Regression")
linear_regression_parser.add_argument(
    "-t",
    "--type",
    default="homoscedastic",
    metavar="{homoscedastic, heteroscedastic} ",
    dest="type",
    type=str,
    help="""Choose wether to generate a homoscesdastic or heteroscedastic epsilon term"""
)
linear_regression_parser.add_argument(
    "-d",
    "--distribution",
    default="uniform",
    type=str,
    metavar="{uniform, normal} ",
    help="Choose wether to generate normal or uniform distributed dataset"
)
linear_regression_parser.add_argument(
    "-i",
    "--iteration",
    default=None,
    type=int,
    metavar="N_iteration (for simulations)"
)

"""linear_regression_parser.add_argument(
    "-n",
    "--neurons",
    default=1024,
    type=int,
    metavar="Number of neurons per hidden layer"
)

linear_regression_parser.add_argument(
    "-l",
    "--layers",
    default=1,
    type=int,
    metavar="Number of hidden layers"
)"""
linear_regression_parser.add_argument(
    "-c",
    "--convergence_threshold",
    default=0.00001,
    type=float,
    metavar="Convergence Threshold of Backfitting algorithm"
)

linear_regression_parser.set_defaults(family='gaussian')

logistic_regression_parser = subparsers.add_parser(name='logistic', help="Logistic Regression")
logistic_regression_parser.add_argument(
    "-d",
    "--distribution",
    default="uniform",
    type=str,
    metavar="{uniform, normal} ",
    help="Choose wether to generate normal or uniform distributed dataset"
)

logistic_regression_parser.add_argument(
    "-i",
    "--iteration",
    default=None,
    type=int,
    metavar="N_iteration (for simulations)"
)

logistic_regression_parser.add_argument(
    "-c",
    "--convergence_threshold",
    default=0.00001,
    type=float,
    metavar="Convergence Threshold of backfitting algorithm"
)

logistic_regression_parser.add_argument(
    "-a",
    "--delta_threshold",
    default=0.01,
    type=float,
    metavar="Convergence Threshold of LS algorithm"
)

logistic_regression_parser.set_defaults(family='binomial')

if __name__ == "__main__":
    
    """ 
    LINEAR REGRESSION: 
        python main.py linear -t {homoscedastic, heteroscedastic} -d {uniform, normal}
    LOGISTIC REGRESSION: 
        python main.py logistic -d {uniform, normal}
    """
    
    args = parser.parse_args()
    variables = vars(args)
    
    type = variables.get("type", None)
    distribution = variables["distribution"]
    family = variables["family"]    # gaussian / binomial
    iteration = variables["iteration"]

    conv_threshold = variables.pop("convergence_threshold", 0.01)
    delta_threshold = variables.pop("delta_threshold", 0.00001)

    distribution = variables["distribution"]

    variables.pop("iteration")
    print(variables)
    
    if iteration is not None:
        rel_path = "./results/{0}".format(iteration)
        path = os.path.normpath(os.path.abspath(rel_path))
        #add iteration
        if not os.path.exists(path):
            os.mkdir(path)

    else:
        rel_path = "./results/"
        path = os.path.normpath(os.path.abspath(rel_path))
        
    # add exec type
    data_type_path = "_".join(list(variables.values())) 
    path = path + "/" + data_type_path
    if not os.path.exists(path):
        os.mkdir(path)

    """ Load dataset """
    try:
        X_train = pd.read_csv("./dataset/{0}/X_train.csv".format(data_type_path), index_col=0).reset_index(drop=True)
        fs_train = pd.read_csv("./dataset/{0}/fs_train.csv".format(data_type_path), index_col=0).reset_index(drop=True)
        y_train = pd.read_csv("./dataset/{0}/y_train.csv".format(data_type_path), index_col=0).reset_index(drop=True).squeeze()

        X_test = pd.read_csv("./dataset/{0}/X_test.csv".format(data_type_path), index_col=0).reset_index(drop=True)
        fs_test = pd.read_csv("./dataset/{0}/fs_test.csv".format(data_type_path), index_col=0).reset_index(drop=True)
        y_test = pd.read_csv("./dataset/{0}/y_test.csv".format(data_type_path), index_col=0).reset_index(drop=True).squeeze()

    except:
        """Not found, generate"""
        X, y, fs = generate_data(nrows=25000, type=type, distribution=distribution, family=family, output_folder=path)
        X_train, X_test, y_train, y_test, fs_train, fs_test = split(X, y, fs)
        
        print("\n Number of elements per class on training set")
        print(pd.DataFrame(np.where(y >= 0.5, 1, 0)).value_counts())
        
    
    if family == "binomial":
        y_train_binomial = np.random.binomial(n=1, p=y_train, size=y_train.shape[0])
        ngam = NeuralGAM(num_inputs = len(X_train.columns), family=family, depth=1, num_units=1024)
        tstart = datetime.now()
        muhat, gs_train = ngam.fit(X_train = X_train, y_train = y_train_binomial, max_iter = 10, convergence_threshold=conv_threshold, delta_threshold=delta_threshold)
        tend = datetime.now()

    else:
        ngam = NeuralGAM(num_inputs = len(X_train.columns), family=family)

        tstart = datetime.now()
        muhat, gs_train = ngam.fit(X_train = X_train, y_train = y_train, max_iter = 10, convergence_threshold=conv_threshold, delta_threshold=delta_threshold)
        tend = datetime.now()
    
    training_seconds = (tend - tstart).seconds
    variables["eta0"] = ngam.eta0
    variables["training_seconds"] = training_seconds

    y_pred = ngam.predict(X_test)
    fs_pred = ngam.get_partial_dependencies(X_test)

    err = mean_squared_error(y_train, ngam.y)
    pred_err = mean_squared_error(y_test, y_pred)
    variables["err"] = err
    variables["err_test"] = pred_err
    
    """ SAVE DATASET """
    if not os.path.exists("./dataset/{0}/X_train.csv".format(data_type_path)):
        pd.DataFrame(X_train).to_csv("./dataset/{0}/X_train.csv".format(data_type_path))
        pd.DataFrame(y_train).to_csv("./dataset/{0}/y_train.csv".format(data_type_path))
        pd.DataFrame(fs_train).to_csv("./dataset/{0}/fs_train.csv".format(data_type_path))
        pd.DataFrame(X_test).to_csv("./dataset/{0}/X_test.csv".format(data_type_path))
        pd.DataFrame(y_test).to_csv("./dataset/{0}/y_test.csv".format(data_type_path))
        pd.DataFrame(fs_test).to_csv("./dataset/{0}/fs_test.csv".format(data_type_path))

    x_list = [X_train, X_train]
    fs_list = [fs_train, gs_train]
    legends = ["real", "estimated_training"]
    vars = variables
    vars["err"] = round(err, 4)
    plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title=vars, output_path=path + "/fs_training.png")
    x_list = [X_test, X_test]
    fs_list = [fs_test, fs_pred]
    legends = ["real_test", "estimated_test"]
    vars = variables
    vars["err"] = round(pred_err, 4)
    plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title=vars, output_path=path + "/fs_test.png")
    

    """ SAVE RESULTS"""
    pd.DataFrame(fs_pred).to_csv(path + "/fs_test_estimated.csv")
    pd.DataFrame(y_pred).to_csv(path + "/y_pred.csv")
   
    pd.DataFrame(gs_train).to_csv(path + "/fs_train_estimated.csv")  
    pd.DataFrame.from_dict(variables, orient="index").transpose().to_csv(path + "/variables.csv", index=False)