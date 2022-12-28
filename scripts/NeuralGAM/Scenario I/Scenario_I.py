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
linear_regression_parser.add_argument(
    "-o",
    "--output",
    default="results",
    dest="output",
    type=str,
    help="""Output folder"""
)

linear_regression_parser.add_argument(
    "-c",
    "--convergence_threshold",
    default=0.00001,
    type=float,
    metavar="Convergence Threshold of backfitting algorithm"
)

linear_regression_parser.add_argument(
    "-a",
    "--delta_threshold",
    default=0.01,
    type=float,
    metavar="Convergence Threshold of LS algorithm"
)

linear_regression_parser.add_argument(
    "-u",
    "--units",
    default=1024,
    type=str,
    dest="units",
    help="""Number of hidden units (i.e. 1024). If a list of values is provided, the neural network will add one hidden layer witch each number of units [1024, 512, 256] will generate a DNN with three hidden layers"""
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
logistic_regression_parser.add_argument(
    "-u",
    "--units",
    default=1024,
    type=str,
    dest="units",
    help="""Number of hidden units (i.e. 1024). If a list of values is provided, the neural network will add one hidden layer witch each number of units [1024, 512, 256] will generate a DNN with three hidden layers"""
)

logistic_regression_parser.add_argument(
    "-o",
    "--output",
    default="results",
    dest="output",
    type=str,
    help="""Output folder"""
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
    
    print(variables)
    type = variables.get("type", None)
    distribution = variables["distribution"]
    family = variables["family"]    # gaussian / binomial
    iteration = variables["iteration"]
    output_folder = variables.pop("output", "results")
    
    conv_threshold = variables.pop("convergence_threshold", 0.01)
    delta_threshold = variables.pop("delta_threshold", 0.00001)
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
    data_type_path = "_".join(list(variables.values()))
    path = path + "/" + data_type_path
    if not os.path.exists(path):
        os.makedirs(path)

    """ Load dataset """

    print(f"Using {data_type_path}")
    try:
        X_train = pd.read_csv("./dataset/{0}/X_train.csv".format(data_type_path), index_col=0).reset_index(drop=True)
        fs_train = pd.read_csv("./dataset/{0}/fs_train.csv".format(data_type_path), index_col=0).reset_index(drop=True)
        y_train = pd.read_csv("./dataset/{0}/y_train.csv".format(data_type_path), index_col=0).reset_index(drop=True).squeeze()

        X_test = pd.read_csv("./dataset/{0}/X_test.csv".format(data_type_path), index_col=0).reset_index(drop=True)
        fs_test = pd.read_csv("./dataset/{0}/fs_test.csv".format(data_type_path), index_col=0).reset_index(drop=True)
        y_test = pd.read_csv("./dataset/{0}/y_test.csv".format(data_type_path), index_col=0).reset_index(drop=True).squeeze()
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        
    
    if family == "binomial":
        
        y_train_binomial = np.random.binomial(n=1, p=y_train, size=y_train.shape[0])
        y_test_binomial =  np.random.binomial(n=1, p=y_test, size=y_test.shape[0])
        ngam = NeuralGAM(num_inputs = len(X_train.columns), family=family, num_units=units)
        tstart = datetime.now()
        muhat, gs_train, eta = ngam.fit(X_train = X_train, 
                                y_train = y_train_binomial, 
                                max_iter_ls = 10, 
                                bf_threshold=conv_threshold,
                                ls_threshold=delta_threshold,
                                max_iter_backfitting=10)
        tend = datetime.now()

    else:
        ngam = NeuralGAM(num_inputs = len(X_train.columns), family=family, num_units=units)
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

    y_pred, eta_pred = ngam.predict(X_test)
    fs_pred = ngam.get_partial_dependencies(X_test)

    if family == "binomial":
        eta_train =  np.log(y_train/(1-y_train))
        eta_test = np.log(y_test/(1-y_test))
        err = mean_squared_error(eta_train, ngam.eta)
        pred_err = mean_squared_error(eta_test, eta_pred)

    else:
        err = mean_squared_error(y_train, ngam.eta)
        pred_err = mean_squared_error(y_test, eta_pred)
    
    variables["err"] = err
    variables["err_test"] = pred_err
    variables["err_rmse"] = np.sqrt(err)
    variables["err_test_rmse"] = np.sqrt(pred_err)
    print("Done")
    print(variables)
    
    if family == "binomial":
        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report, confusion_matrix
        threshold = youden(y_test_binomial, y_pred)
        y_bin = np.where(y_pred >= threshold, 1, 0)
        pr, rec, f1, support = precision_recall_fscore_support(y_test_binomial, y_bin)

        auc = roc_auc_score(y_test_binomial, y_pred)
        print("Achieved AUC {0}".format(auc))

        variables["auc_roc"] = auc 
        variables["precission"] = pr
        variables["recall"] = rec
        variables["f1"] = f1
        variables["threshold"] = threshold
        metrics = classification_report(y_test_binomial, y_bin, output_dict=True)
        matrix = confusion_matrix(y_test_binomial, y_bin, normalize="true")
        tn, fp, fn, tp = matrix.ravel()
        variables["tp"] = tp
        variables["fp"] = fp
        variables["fn"] = fn
        variables["tp"] = tp
       
        pd.DataFrame(y_train_binomial).to_csv(path + "/y_train_binomial.csv")
        pd.DataFrame(y_test_binomial).to_csv(path + "/y_test_binomial.csv")
        pd.DataFrame(y_bin).to_csv(path + "/y_pred_binomial.csv") 
        

    """ SAVE RESULTS"""
    pd.DataFrame(fs_pred).to_csv(path + "/fs_test_estimated.csv")
    pd.DataFrame(y_pred).to_csv(path + "/y_pred.csv")
    
    pd.DataFrame(gs_train).to_csv(path + "/fs_train_estimated.csv")  
    pd.DataFrame.from_dict(variables, orient="index").transpose().to_csv(path + "/variables.csv", index=False)