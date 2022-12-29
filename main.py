import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    default="./dataset/homoscedastic_uniform_gaussian",
    dest="input",
    type=str,
    help="""Input folder - place here X_train, y_train, etc..."""
)
parser.add_argument(
    "-o",
    "--output",
    default="results",
    dest="output",
    type=str,
    help="""Output folder - results and model"""
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
    "-f",
    "--family",
    default="gaussian",
    type=str,
    metavar="Distribution Family. Use gaussian for LINEAR REGRESSION and binomial for LOGISTIC REGRESSION"
)

parser.add_argument(
    "-l",
    "--lr",
    dest="lr",
    default=0.001,
    type=float,
    metavar="Learning Rate. Defaults to 0.001"
)

parser.add_argument(
    "-c",
    "--bf_threshold",
    default=0.00001,
    type=float,
    metavar="Convergence Threshold of backfitting algorithm. Defaults to 10e-5"
)
parser.add_argument(
    "-d",
    "--ls_threshold",
    default=0.01,
    type=float,
    metavar="Convergence Threshold of LS algorithm. Defaults to 0.1"
)
parser.add_argument(
    "-ls",
    "--maxiter_ls",
    default=10,
    dest="ls",
    type=int,
    metavar="Max iterations of LS algorithm. Defaults to 10."
)
parser.add_argument(
    "-bf",
    "--maxiter_bf",
    dest="bf",
    default=10,
    type=int,
    metavar="Max iterations of Backfitting algorithm. Defaults to 10"
)
if __name__ == "__main__":
    
    args = parser.parse_args()
    variables = vars(args)

    print(f'\n\n{variables}\n\n')
    
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from src.NeuralGAM.ngam import NeuralGAM, plot_partial_dependencies
    import pandas as pd

    units = [int(item) for item in variables["units"].split(',')]
    lr = variables["lr"]

    output_path = os.path.normpath(os.path.abspath(os.path.join("./", variables["output"])))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_path = os.path.normpath(os.path.abspath(os.path.join("./", variables["input"])))

    """ Load dataset -- if you want to preprocess or select some features, do it here"""
    try:
        X_train = pd.read_csv(os.path.join(input_path, "X_train.csv"), index_col=0).reset_index(drop=True)
        y_train = pd.read_csv(os.path.join(input_path, "y_train.csv"), index_col=0).reset_index(drop=True).squeeze()

        X_test = pd.read_csv(os.path.join(input_path, "X_test.csv"), index_col=0).reset_index(drop=True)
        y_test = pd.read_csv(os.path.join(input_path, "y_test.csv"), index_col=0).reset_index(drop=True).squeeze()
    except Exception as e:
        print("Failed to load data from {0}: {1}".format(input_path, e))
        exit(-1)

    print("Startint NeuralGAM training  with {0} rows...".format(X_train.shape[0]))
    ngam = NeuralGAM(num_inputs = len(X_train.columns), family=variables["family"], num_units=units, learning_rate=lr)

    if variables["family"] != "gaussian":
        """ Ensure y_test / y_train are proper labels..."""
        if not np.logical_or(y_test == 0, y_test == 1).all() and not np.logical_or(y_train == 0, y_train == 1).all():
            raise Exception("To use Logistic Regression you must provide train/test labels in the discrete set {0,1}")

    muhat, fs_train, eta = ngam.fit(X_train = X_train, 
                                y_train = y_train, 
                                max_iter_ls = variables["ls"], 
                                bf_threshold=variables["bf_threshold"],
                                ls_threshold=variables["ls_threshold"],
                                max_iter_backfitting=variables["bf"])
    err = mean_squared_error(y_train, ngam.y)
    print("Trainign done... MSE_train = {0}".format(str(err)))
    results = dict()

    results["muhat"] = str(muhat)
    results["MSE_train"] = str(err)

    print("Starting Predict...")
    y_pred, eta_pred = ngam.predict(X_test)

    if variables["family"] == "gaussian":
        pred_err = mean_squared_error(y_test, eta_pred)
        results["MSE_test"] = str(pred_err)
        print("Predict done... MSE_test = {0}".format(str(pred_err)))
    else:
        """ Binomial scenario. Compute AUC/ROC"""
        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report, confusion_matrix
        from src.utils.utils import youden
        
        threshold = youden(y_test, y_pred)
        y_bin = np.where(y_pred >= threshold, 1, 0)
        pr, rec, f1, support = precision_recall_fscore_support(y_test, y_bin)
        auc = roc_auc_score(y_test, y_pred)
        print("Achieved AUC {0}".format(auc))

        variables["auc_roc"] = auc 
        variables["threshold"] = threshold
        metrics = classification_report(y_test, y_bin, output_dict=True)
        matrix = confusion_matrix(y_test, y_bin, normalize="true")
        tn, fp, fn, tp = matrix.ravel()
        variables["tp"] = tp
        variables["fp"] = fp
        variables["fn"] = fn
        variables["tp"] = tp
        pd.DataFrame(y_bin).to_csv(output_path + "/y_pred_binomial.csv")
        
    print("Obtaining Partial Dependence Plots...")
    fs_pred = ngam.get_partial_dependencies(X_test)
        
    plot_partial_dependencies(x=X_train, fs=fs_train, title="Training Partial Dependence Plot", output_path=output_path + "/fs_train.png")
    
    """ SAVE RESULTS"""
    pd.DataFrame(fs_pred).to_csv(output_path + "/fs_test_estimated.csv")
    pd.DataFrame(y_pred).to_csv(output_path + "/y_pred.csv")
    pd.DataFrame(fs_train).to_csv(output_path + "/fs_train_estimated.csv")  
    pd.DataFrame(eta).to_csv(output_path + "/eta.csv")  
    pd.DataFrame.from_dict(results, orient="index").transpose().to_csv(output_path + "/results.csv", index=False)

    print(f"\n\n{variables}\n\n")