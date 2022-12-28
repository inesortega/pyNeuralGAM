import numpy as np
import pandas as pd
import os
import sys
import argparse 

if __name__ == "__main__":
    # get random number of iterations on range [1, 1000]

    list_of_arguments = sys.argv

    if (len(list_of_arguments) < 2):
        print("You must provide an input folder: python stats.py FOLDER")
    
    input_folder = list_of_arguments[1]
    nam = list_of_arguments[2]

    if nam == "nam":
        is_nam = True
    else:
        is_nam = False
        
    rel_path = "./"
    path = os.path.join(os.path.normpath(os.path.abspath(rel_path)), input_folder)

    print("generating statistics reading data from {0}".format(path))

    types = os.listdir(os.path.join(path, os.listdir(path)[0]))# get subdirs, for instance   ["homoscedastic_uniform_gaussian", "heteroscedastic_uniform_gaussian", "uniform_binomial"]
    
    dirs = os.listdir(path) # get N iterations
    dirs = [dir for dir in dirs if dir.isnumeric()]

    all_pred = np.zeros(len(dirs), dtype=object)   # for 0-1 loss you might need int64

    for type in types:
        try:
            # Create output dir
            output_path = os.path.join(path, type)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            
            # obtain pred to compute mean bias / variance
            y_test = pd.read_csv("./dataset/{0}/y_test.csv".format(type), index_col=0).reset_index(drop=True)["0"].values
            
            if type == "uniform_binomial":
                y_test = np.log(y_test/(1-y_test))
                
            err_train = pd.DataFrame()
            err_pred = pd.DataFrame()
            training_seconds = pd.DataFrame()
            all_pred = pd.DataFrame()

            for j in dirs:
                variables = pd.read_csv(os.path.join(path, str(j), type, "variables.csv"), index_col=0).reset_index(drop=True)

                try:
                    variables_training = pd.read_csv(os.path.join(path, str(j), type, "variables_training.csv"), index_col=0).reset_index(drop=True)
                    training_seconds = pd.concat([training_seconds, variables_training["training_seconds"]], axis=0)
                except:
                    training_seconds = pd.concat([training_seconds, variables["training_seconds"]], axis=0)
                
                y_pred = pd.read_csv(os.path.join(path, str(j), type, "y_pred.csv"), index_col=0).reset_index(drop=True)
                
                if type == "uniform_binomial" and not is_nam:
                    y_pred = np.log(y_pred/(1-y_pred))

                all_pred = pd.concat([all_pred, y_pred], axis=1)
                from sklearn.metrics import mean_squared_error
                mse = mean_squared_error(y_test, y_pred)
                err_pred = pd.concat([err_pred, pd.Series(mse)], axis=0)

            err_pred.to_csv("{0}/err_pred.csv".format(output_path), index=False)
            training_seconds.to_csv("{0}/training_seconds.csv".format(output_path), index=False)

            
            all_pred = all_pred.to_numpy()
            
            # Bias: for each x, mean of 1000 estimations of x_i - y_test (averaged)
            bias = (all_pred.mean(axis=1) - y_test).mean()
            # Variance: variance of 1000 estimations, for each x_i (averaged)
            var = all_pred.var(axis=1).mean()

            mean_values = {}
            mean_values["err_pred"]  = err_pred.mean()[0]
            mean_values["training_seconds"]  = training_seconds.mean()[0]
            mean_values["bias"] = bias
            mean_values["var"] = var
            
            pd.DataFrame.from_dict(mean_values, orient="index").transpose().to_csv("{0}/mean_metrics.csv".format(output_path), index=False)

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            print("error on it type={0}".format(type))
            print(range)
