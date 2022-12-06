from inspect import trace
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use('Agg') 
import os
import seaborn as sns
import sys

if __name__ == "__main__":

    def inverse_min_max_scaler(x, min_val, max_val):
        return (x + 1)/2 * (max_val - min_val) + min_val    

    def preprocess_X_google(X):
        NUM_FEATURES = X.shape[1]
        SINGLE_FEATURES = np.split(X, NUM_FEATURES, axis=1)
        UNIQUE_FEATURES = [np.unique(x, axis=0) for x in SINGLE_FEATURES]
        
        UNIQUE_FEATURES_ORIGINAL = {}

        col_min_max = {}
        for col in X.columns:
            unique_vals = X[col].unique()
            col_min_max[col] = (np.min(unique_vals), np.max(unique_vals))

        for i, col in enumerate(X.columns):
            min_val, max_val = col_min_max[col]
            UNIQUE_FEATURES_ORIGINAL[col] = inverse_min_max_scaler(
                UNIQUE_FEATURES[i][:, 0], min_val, max_val)

        return pd.DataFrame(UNIQUE_FEATURES_ORIGINAL)

    def compute_mean_estimations(X):
        f_list = list()
        mean_estimations = pd.DataFrame()
        q975 = pd.DataFrame()
        q025 = pd.DataFrame()
        for i in X.columns:
            f_list.append(pd.DataFrame())

        dirs = os.listdir(path)

        for dir in dirs:
            try:
                fs_estimated_i = pd.read_csv(os.path.join(path, dir, type, "fs_train_estimated.csv"), index_col=0).reset_index(drop=True)
            except:
                continue
            for i, f in enumerate(fs_estimated_i.columns):
                f_list[i] = pd.concat([f_list[i], fs_estimated_i[f]], axis=1)

        for i, f in enumerate(f_list):
            if is_nam:
                f_list[i] = f_list[i] - f_list[i].mean()
            mean_estimations[str(i)] = f_list[i].mean(axis=1)
            q975[str(i)] = f_list[i].quantile(0.975, axis=1)
            q025[str(i)] = f_list[i].quantile(0.025, axis=1)
        
        return mean_estimations.reset_index(drop=True), q975.reset_index(drop=True), q025.reset_index(drop=True)

    list_of_arguments = sys.argv
    if (len(list_of_arguments) < 2):
        print("You must provide an input folder: python compute_mean_estimations.py FOLDER")
    
    input_folder = list_of_arguments[1]
    nam = list_of_arguments[2]

    if nam == "nam":
        is_nam = True
    else:
        is_nam = False

    print("generating data for plotting from {0}...".format(input_folder))

    rel_path = "./"
    path = os.path.normpath(os.path.abspath(os.path.join("./", input_folder)))

    types = os.listdir(os.path.join(path, os.listdir(path)[0]))
    for type in types:
        try:
            output_path = os.path.join(path, type)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            X_train = pd.read_csv("./dataset/{0}/X_train.csv".format(type), index_col=0).reset_index(drop=True)
            fs_train = pd.read_csv("./dataset/{0}/fs_train.csv".format(type), index_col=0).reset_index(drop=True)
            
            # center theoretical fs for plotting
            fs_train = fs_train - fs_train.mean()

            print("Generating " + type + " plots")
            
            mean_estimations, q975, q025 = compute_mean_estimations(X_train)

            """if is_nam:
                X_train_prec = preprocess_X_google(X_train)
                X_train_prec = X_train_prec.reset_index(drop=True)
                X_train_prec.to_csv(("./dataset/{0}/X_train_postprocess.csv".format(type)))
            """
            
            mean_estimations.to_csv(os.path.join(output_path, "mean_estimation") + ".csv")
            q975.to_csv(os.path.join(output_path, "q975") + ".csv")
            q025.to_csv(os.path.join(output_path, "q025") + ".csv")
            X_train.to_csv(os.path.join(output_path, "X_train") + ".csv")
            fs_train.to_csv(os.path.join(output_path, "fs_train") + ".csv")

        except Exception as e:
            print(e) 
            import traceback
            traceback.print_exc()
            print("error on it type={0}".format(type))
            print(range)
