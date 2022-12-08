import numpy as np
import pandas as pd
import os
import sys

if __name__ == "__main__":
    # get random number of iterations on range [1, 1000]

    list_of_arguments = sys.argv

    if (len(list_of_arguments) < 2):
        print("You must provide an input folder: python stats.py FOLDER")
    
    input_folder = list_of_arguments[1]

    rel_path = "./"
    path = os.path.join(os.path.normpath(os.path.abspath(rel_path)), input_folder)

    print("generating statistics reading data from {0}".format(path))

    types = [x[1] for x in os.walk(os.path.join(path, '1'))][0] # get subdirs, for instance   ["homoscedastic_uniform_gaussian", "heteroscedastic_uniform_gaussian", "uniform_binomial"]
    
    dirs = os.listdir(path) # get N iterations
    dirs = [dir for dir in dirs if dir.isnumeric()]

    all_pred = np.zeros(len(dirs), dtype=object)   # for 0-1 loss you might need int64

    for type in types:
        try:
            # Create output dir
            output_path = os.path.join(path, type)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            
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
                
                err_pred = pd.concat([err_pred, variables["err_test"]], axis=0)
                y_pred = pd.read_csv(os.path.join(path, str(j), type, "y_pred.csv"), index_col=0).reset_index(drop=True)
                all_pred = pd.concat([all_pred, y_pred], axis=1)

            err_pred.to_csv("{0}/err_pred.csv".format(output_path), index=False)
            training_seconds.to_csv("{0}/training_seconds.csv".format(output_path), index=False)

            # obtain pred to compute mean bias / variance
            y_test = pd.read_csv("./dataset/{0}/y_test.csv".format(type), index_col=0).reset_index(drop=True)["0"].values

            all_pred = all_pred.to_numpy()
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
