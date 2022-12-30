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
        "font.family": "serif"}

matplotlib.rcParams['agg.path.chunksize'] = 10000
plt.rcParams.update(params)
sns.set(rc={"figure.dpi":550, 'savefig.dpi':550})

if __name__ == "__main__":
    
    list_of_arguments = sys.argv
    if (len(list_of_arguments) < 2):
        print("You must provide an input folder: python plots.py FOLDER")
    
    input_folder = list_of_arguments[1]

    print("generating plots from {0}...".format(input_folder))

    rel_path = "./"
    path = os.path.normpath(os.path.abspath(os.path.join("./", input_folder)))
    
    types = os.listdir(os.path.join(path))

    for type in types:
        try:
            output_path = os.path.join(path, type)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            X_train = pd.read_csv("./dataset/{0}/X_train.csv".format(type), index_col=0).reset_index(drop=True)
            fs = pd.read_csv("./dataset/{0}/fs_train.csv".format(type), index_col=0).reset_index(drop=True)
            
            # center theoretical fs for plotting
            fs = fs - fs.mean()

            print("Generating " + type + " plots")
            
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8.27*5,3.896*5))  # A4 width, A4/3 height (proportional) -- For squared use (25,20)

            # Set tick font size
            for ax in axs:
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(30)

            mean_estimations = pd.read_csv(os.path.join(path, type, "mean_estimation.csv"), index_col=0).reset_index(drop=True)
            q975 = pd.read_csv(os.path.join(path, type, "q975.csv"), index_col=0).reset_index(drop=True)
            q025 = pd.read_csv(os.path.join(path, type, "q025.csv"), index_col=0).reset_index(drop=True)
            
            for i, term in enumerate(X_train.columns):
                
                data = pd.DataFrame()
                
                # Plot theoretical f(x)
                data['x'] = X_train[str(i)]
                data['y'] = fs[str(i)]
                sns.lineplot(data = data, x='x', y='y', color='royalblue', ax=axs[i])
                
                #Plot mean estimation
                data = pd.DataFrame()
                data['x'] = X_train[str(i)]
                data['y'] = mean_estimations[str(i)]
                sns.lineplot(data = data, x='x', y='y', ax=axs[i], color='mediumseagreen', linestyle='--', alpha=0.7)
                
                # draw quantiles
                data['q975'] = q975[str(i)]
                data['q025'] = q025[str(i)]

                sns.lineplot(data = data, x='x', y='q975', color='coral', alpha=0.5, ax=axs[i])
                axs[i].lines[-1].set_linestyle('--')
                sns.lineplot(data = data, x='x', y='q025', color='coral', alpha=0.5, ax=axs[i])
                axs[i].lines[-1].set_linestyle('--')
                
                axs[i].set_xlabel(f"$X_{i+1}$", fontsize=30)
                axs[i].set_ylabel(f"$f(x_{i+1})$", fontsize=30)

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, type) + ".png", dpi=550, bbox_inches = "tight")
            plt.clf()
            
        except Exception as e:
            print(e) 
            import traceback
            traceback.print_exc()
            print("error on it type={0}".format(type))