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

plt.style.use('seaborn')

params = {"axes.linewidth": 2,
        "font.size": 30, 
        "font.family": "serif",
        "axes.labelsize": 30}

matplotlib.rcParams['agg.path.chunksize'] = 10000
plt.rcParams.update(params)

if __name__ == "__main__":
    
    list_of_arguments = sys.argv
    if (len(list_of_arguments) < 2):
        print("You must provide an input folder: python plots.py FOLDER")
    
    input_folder = list_of_arguments[1]

    print("generating plots from {0}...".format(input_folder))

    rel_path = "./"
    path = os.path.normpath(os.path.abspath(os.path.join("./", input_folder)))
    
    types = os.listdir(path)
    
    for type in types:
        try:
            output_path = os.path.join(path, type)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            X = pd.read_csv(os.path.join(path, type, "X_grid.csv"), index_col=0).reset_index(drop=True)
            f = pd.read_csv(os.path.join(path, type, "fs_grid.csv"), index_col=0).reset_index(drop=True)
            
            print("Generating " + type + " plots")
            
            for i,var in enumerate(X.columns):
            
                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
                
                for label in (axs.get_xticklabels() + axs.get_yticklabels()):
                    label.set_fontsize(30)
                
                data = pd.DataFrame()
                data['x'] = X.iloc[:,i]
                data['y'] = f.iloc[:,i]
                sns.lineplot(data = data, x='x', y='y', color='royalblue', linewidth=5)
                
                axs.set_xlabel(var)
                axs.set_ylabel(f"$\hat f(x)$")
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, type + "_" + var) + ".png", dpi=500, bbox_inches = "tight")
                plt.clf()
            
        except Exception as e:
            print(e) 
            import traceback
            traceback.print_exc()
            print("error on it type={0}".format(type))