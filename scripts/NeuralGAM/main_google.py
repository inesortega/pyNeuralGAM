import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from src.utils.utils import generate_data, split, youden
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
linear_regression_parser.add_argument(
    "-o",
    "--output",
    default="results",
    dest="output",
    type=str,
    help="""Output folder"""
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
    
    type = variables.get("type", None)
    distribution = variables["distribution"]
    family = variables["family"]    # gaussian / binomial
    iteration = variables["iteration"]
    output_folder = variables.pop("output", "results")
    
    conv_threshold = variables.pop("convergence_threshold", 0.01)
    delta_threshold = variables.pop("delta_threshold", 0.00001)

    variables.pop("iteration")
    print(variables)
    
    if iteration is not None:
        rel_path = "./{0}/{1}".format(output_folder, iteration)
        path = os.path.normpath(os.path.abspath(rel_path))
        #add iteration
        if not os.path.exists(path):
            os.mkdir(path)

    else:
        rel_path = "./{0}/".format(output_folder)
        path = os.path.normpath(os.path.abspath(rel_path))
        
    # add exec type
    #data_type_path = "_".join(list(variables.values()))
    
    data_type_path = "google"
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
    except Exception as e:
        """Not found, generate"""

        print("Dataset not found!")
        X, y, fs = generate_data(nrows=25000, type=type, distribution=distribution, family=family, output_folder=path)
        X_train, X_test, y_train, y_test, fs_train, fs_test = split(X, y, fs)
        
        print("\n Number of elements per class on training set")
        print(pd.DataFrame(np.where(y >= 0.5, 1, 0)).value_counts())
        
    
    if family == "binomial":
        y_train_binomial = np.random.binomial(n=1, p=y_train, size=y_train.shape[0])
        y_test_binomial =  np.random.binomial(n=1, p=y_test, size=y_test.shape[0])
        ngam = NeuralGAM(num_inputs = len(X_train.columns), family=family, depth=1, num_units=1024)
        tstart = datetime.now()
        muhat, gs_train = ngam.fit(X_train = X_train, y_train = y_train_binomial, max_iter = 10, convergence_threshold=conv_threshold, delta_threshold=delta_threshold)
        tend = datetime.now()

    else:
        ngam = NeuralGAM(num_inputs = len(X_train.columns), family=family, num_units=1024, depth=3)
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
    variables["err_rmse"] = np.sqrt(err)
    variables["err_test_rmse"] = np.sqrt(pred_err)
    print("Done")
    print(variables)

    """ SAVE DATASET """
    if not os.path.exists("./dataset/{0}/X_train.csv".format(data_type_path)):
        pd.DataFrame(X_train).to_csv("./dataset/{0}/X_train.csv".format(data_type_path))
        pd.DataFrame(y_train).to_csv("./dataset/{0}/y_train.csv".format(data_type_path))
        pd.DataFrame(fs_train).to_csv("./dataset/{0}/fs_train.csv".format(data_type_path))
        pd.DataFrame(X_test).to_csv("./dataset/{0}/X_test.csv".format(data_type_path))
        pd.DataFrame(y_test).to_csv("./dataset/{0}/y_test.csv".format(data_type_path))
        pd.DataFrame(fs_test).to_csv("./dataset/{0}/fs_test.csv".format(data_type_path))
  
    legends = ["real_test", "estimated_test"]
    vars = variables
    vars["err"] = round(pred_err, 4)
    
    #fs_test = fs_test - fs_test.mean()
    #plot_multiple_partial_dependencies(x_list=[X_test, X_test], f_list=[fs_test, fs_pred], legends=legends, title=vars, output_path=path + "/fs_test.png")

    
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