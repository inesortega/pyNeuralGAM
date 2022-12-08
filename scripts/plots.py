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
    
    nam = list_of_arguments[2]

    if nam == "nam":
        is_nam = True

    else:
        is_nam = False

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
            
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25,20))
            
            # Set tick font size
            for ax in axs:
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(19)

            mean_estimations = pd.read_csv(os.path.join(path, type, "mean_estimation.csv"), index_col=0).reset_index(drop=True)
            q975 = pd.read_csv(os.path.join(path, type, "q975.csv"), index_col=0).reset_index(drop=True)
            q025 = pd.read_csv(os.path.join(path, type, "q025.csv"), index_col=0).reset_index(drop=True)

            for i, term in enumerate(X_train.columns):
                
                data = pd.DataFrame()
                
                # Plot theoretical f(x)
                data['x'] = X_train[str(i)]
                data['y'] = fs_train[str(i)]
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
                
            axs[0].set_xlabel(f"$X_1$")
            axs[0].set_ylabel(f"$f(x_1)$")
            
            axs[1].set_xlabel(f"$X_2$")
            axs[1].set_ylabel(f"$f(x_2)$")

            axs[2].set_xlabel(f"$X_3$")
            axs[2].set_ylabel(f"$f(x_3)$")

            """theoretical_patch = mpatches.Patch(color='royalblue', label='Theoretical f(x)')
            learned_patch = mpatches.Patch(color='mediumseagreen', label='Learned $\hat{f}(x)$')
            quantiles = mpatches.Patch(color='coral', label='2.5% and 97.5% simulation quantiles')
            
            plt.subplots_adjust(bottom=0.85)
            fig.legend(handles=[theoretical_patch, learned_patch, quantiles], loc='lower center', ncol=5,
            fancybox=True, shadow=True)
            """
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, type) + ".png", dpi=500, bbox_inches = "tight")
            
        except Exception as e:
            print(e) 
            import traceback
            traceback.print_exc()
            print("error on it type={0}".format(type))