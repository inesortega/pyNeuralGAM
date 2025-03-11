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
    metavar="Convergence Threshold of LS algorithm. Defaults to 0.01"
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
    
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from src.NeuralGAM.ngam_oneVsRest import NeuralGAM
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
        fs_train = pd.read_csv(os.path.join(input_path, "fs_train.csv"), index_col=0).reset_index(drop=True)
        
        # center theoretical fs for plotting
        fs_train = fs_train - fs_train.mean()

        X_test = pd.read_csv(os.path.join(input_path, "X_test.csv"), index_col=0).reset_index(drop=True)
        y_test = pd.read_csv(os.path.join(input_path, "y_test.csv"), index_col=0).reset_index(drop=True).squeeze()
        fs_test = pd.read_csv(os.path.join(input_path, "fs_test.csv"), index_col=0).reset_index(drop=True)
        fs_test = fs_test - fs_test.mean()

        #Convert to 0/1 with probability y_train
        if variables["family"] == "binomial":
            y_train = np.random.binomial(n=1, p=y_train, size=y_train.shape[0])
            y_test =  np.random.binomial(n=1, p=y_test, size=y_test.shape[0])
        
    except Exception as e:
        print("Failed to load data from {0}: {1}".format(input_path, e))
        exit(-1)

    print("Startint NeuralGAM training  with {0} rows...".format(X_train.shape[0]))
    ngam = NeuralGAM(p_terms = ["1"], np_terms=["0", "2"], family=variables["family"], num_units=units, learning_rate=lr)

    if variables["family"] != "gaussian":
        """ Ensure y_test / y_train are proper labels..."""
        if not (np.logical_or(y_train == 0, y_train == 1).all()):
            raise Exception("To use Logistic Regression you must provide train labels in the discrete set {0,1}")
    
    import time
    start_time = time.time()
    muhat, fs_train_est, eta = ngam.fit(X_train = X_train, 
                                y_train = y_train, 
                                max_iter_ls = variables["ls"], 
                                bf_threshold=variables["bf_threshold"],
                                ls_threshold=variables["ls_threshold"],
                                max_iter_backfitting=variables["bf"],
                                parallel=True)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print("Starting Predict...")
    y_pred = ngam.predict(X_test, type = "response")
    eta_pred = ngam.predict(X_test, type = "link")
    if variables["family"] == "gaussian":
        pred_err = mean_squared_error(y_test, eta_pred)
        variables["MSE_test"] = str(pred_err)
        print("Predict done... MSE_test = {0}".format(str(pred_err)))
    else:
        """ Binomial scenario. Compute AUC/ROC"""
        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report, confusion_matrix
        from src.utils.utils import youden
        
        try:
            """ try to find probabilities on dataset folder..."""
            y_test_prob = pd.read_csv(os.path.join(input_path, "y_test_prob.csv"), index_col=0).reset_index(drop=True).squeeze()
            """ apply link function to probabilities to get eta"""
            eta_test = np.log(y_test_prob/(1-y_test_prob))
            pred_err = mean_squared_error(eta_test, eta_pred)
            variables["MSE_test"] = str(pred_err)
        except:
            pass

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
        variables["tn"] = tn
        pd.DataFrame(y_bin).to_csv(output_path + "/y_pred_binomial.csv")
        
    print("Obtaining Partial Dependence Plots...")
    fs_pred = ngam.predict(X_test, type="terms")
        
    from src.utils.utils import experiments_plot_partial_dependencies
    experiments_plot_partial_dependencies(x_list=[X_train, X_train], f_list=[fs_train, fs_train_est], legends=["true", "estimated"], title="Learnt Training Partial Effects", output_path=output_path + "/fs_train.png")
    
    experiments_plot_partial_dependencies(x_list=[X_test, X_test], f_list=[fs_test, fs_pred], legends=["true", "estimated"], title="Partial Effects obtained in Test", output_path=output_path + "/fs_test.png")


    """ SAVE RESULTS"""
    pd.DataFrame(fs_pred).to_csv(output_path + "/fs_test_estimated.csv")
    pd.DataFrame(y_pred).to_csv(output_path + "/y_pred.csv")
    pd.DataFrame(fs_train_est).to_csv(output_path + "/fs_train_estimated.csv")  
    pd.DataFrame(eta).to_csv(output_path + "/eta.csv")  
    pd.DataFrame.from_dict(variables, orient="index").transpose().to_csv(output_path + "/results.csv", index=False)

    print(f"\n\n{variables}\n\n")