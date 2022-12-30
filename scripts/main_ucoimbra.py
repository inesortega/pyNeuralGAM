import argparse
import os
from random import uniform
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, confusion_matrix
from src.utils.utils import generate_data, split, youden
from src.NeuralGAM.ngam import NeuralGAM
import pandas as pd
from datetime import datetime
parser = argparse.ArgumentParser()

parser.add_argument(
    "-o",
    "--output",
    default="results",
    dest="output",
    type=str,
    help="""Output folder"""
)

parser.add_argument(
    "-i",
    "--input",
    dest="input",
    type=str,
    help="""Output folder"""
)

parser.add_argument(
    "-d",
    "--delta_threshold",
    dest="delta_threshold",
    type=float,
    help="""Local Scoring Convergence Threshold"""
)

if __name__ == "__main__":
    
    args = parser.parse_args()
    variables = vars(args)
    
    output_folder = variables.pop("output", "results")
    data_type_path = variables.pop("input")
    delta_threshold = variables.pop("delta_threshold")
    
    path = os.path.normpath(os.path.abspath(os.path.join("./", output_folder, data_type_path)))
    
    if not os.path.exists(path):
        os.makedirs(path)

    X_train = pd.read_csv("./dataset/{0}/X_train.csv".format(data_type_path), index_col=0).reset_index(drop=True)
    y_train = pd.read_csv("./dataset/{0}/y_train.csv".format(data_type_path), index_col=0).reset_index(drop=True).squeeze()

    X_test = pd.read_csv("./dataset/{0}/X_test.csv".format(data_type_path), index_col=0).reset_index(drop=True)
    y_test = pd.read_csv("./dataset/{0}/y_test.csv".format(data_type_path), index_col=0).reset_index(drop=True).squeeze()

    feature_list = ["pkts", "bytes","dur","sintpkt", "dintpkt"]
    
    X_train = X_train[feature_list]
    X_test = X_test[feature_list]
    
    X_train = X_train.set_axis(feature_list, axis=1, inplace=False)

    y_train = y_train.iloc[X_train.index]
    y_test = y_test.iloc[X_test.index]
    
    variables["conv_threshold"] = 1e-5
    variables["delta_threshold"] = delta_threshold
    variables["num_units"] = [1024]

    ngam = NeuralGAM(num_inputs = len(X_train.columns), family="binomial", num_units=variables["num_units"], learning_rate=0.001)
    tstart = datetime.now()

    print(X_train.shape)
    muhat, gs_train, eta_train = ngam.fit(X_train = X_train, 
                                y_train = y_train, 
                                max_iter_ls = 10, 
                                bf_threshold=10e-5,
                                ls_threshold=delta_threshold,
                                max_iter_backfitting=10)
    tend = datetime.now()

    training_seconds = (tend - tstart).seconds
    variables["eta0"] = ngam.eta0
    variables["training_seconds"] = training_seconds

    y_pred, eta_pred = ngam.predict(X_test)
    fs_pred = ngam.get_partial_dependencies(X_test)

    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
    auc = roc_auc_score(y_test, y_pred)
    print("Achieved AUC {0}".format(auc))
    threshold = youden(y_test, y_pred)
    y_bin = np.where(y_pred >= threshold, 1, 0)
    pd.DataFrame(y_bin).to_csv(path + "/y_bin.csv")  
    
    variables["auc"] = auc 
    variables["threshold"] = threshold
    
    matrix = confusion_matrix(y_test, y_bin, normalize="true")
    print("Normalized Confussion Matrix")
    tn, fp, fn, tp = matrix.ravel()
    variables["tp"] = tp
    variables["fp"] = fp
    variables["fn"] = fn
    variables["tn"] = tn

    print(variables)

    """ SAVE RESULTS"""
    pd.DataFrame(fs_pred).to_csv(path + "/fs_test_estimated.csv")
    pd.DataFrame(y_pred).to_csv(path + "/y_pred.csv")
   
    pd.DataFrame(gs_train).to_csv(path + "/fs_train_estimated.csv")  
    pd.DataFrame.from_dict(variables, orient="index").transpose().to_csv(path + "/variables.csv", index=False)


    """ Generate Plots"""

    # Create grid for each feature with 500 points from a normal distribution
    
    X_grid = pd.DataFrame()
    for i, column in enumerate(X_test.columns):
        col = X_test[column]
        X_grid[i] = np.array(np.random.uniform(low=col.min(), high=col.max(), size=500)).tolist()
        if column == "pkts":
            X_grid[i] = X_grid[i].apply(np.ceil)

    fs_grid = ngam.get_partial_dependencies(X_grid)
    X_grid.columns = feature_list

    fs_grid.to_csv(path + "/fs_grid.csv")
    X_grid.to_csv(path + "/X_grid.csv")