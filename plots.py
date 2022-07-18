from inspect import trace
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use('Agg') 
import os
import seaborn as sns


if __name__ == "__main__":
    
    def compute_mean_estimations(X_train):
        f_list = list()
        mean_estimations = pd.DataFrame()
        q975 = pd.DataFrame()
        q025 = pd.DataFrame()
        for i in X_train.columns:
            f_list.append(pd.DataFrame())

        for j in range(1, RANGE_SIZE+1):
            fs_estimated_i = pd.read_csv("./results/{0}/{1}/fs_test_estimated.csv".format(j, type), index_col=0).reset_index(drop=True)
            
            for i, f in enumerate(f_list):
                f_list[i] = pd.concat([f_list[i], fs_estimated_i[str(i)]], axis=1)

        for i, f in enumerate(f_list):
            mean_estimations[i] = f_list[i].mean(axis=1)
            q975[i] = f_list[i].quantile(0.975, axis=1)
            q025[i] = f_list[i].quantile(0.025, axis=1)
        
        return mean_estimations.reset_index(drop=True), q975.reset_index(drop=True), q025.reset_index(drop=True)


    # get random number of iterations on range [1, 1000]

    print("generating random plots from 100 iterations...")


    rel_path = "./"
    path = os.path.normpath(os.path.abspath(rel_path))
    
    RANGE_SIZE = 1000

    fig, axs = plt.subplots(nrows=1, ncols=1)
    
    plot_types = ["homoscedastic_uniform_gaussian", "heteroscedastic_uniform_gaussian", "uniform_binomial"]

    for type in plot_types:
        try:
            X_train = pd.read_csv("./dataset/{0}/X_train.csv".format(type), index_col=0).reset_index(drop=True)
            fs_train = pd.read_csv("./dataset/{0}/fs_train.csv".format(type), index_col=0).reset_index(drop=True)
            
            X_test = pd.read_csv("./dataset/{0}/X_test.csv".format(type), index_col=0).reset_index(drop=True)
            
            # center theoretical fs for plotting
            fs_train = fs_train - fs_train.mean()

            fig, axs = plt.subplots(nrows=1, ncols=3)

            print("Generating " + type + " plots")
            fig.suptitle("Theoretical training data and estimated functions (1000 iterations) \n " + type, fontsize=10)
            
            mean_estimations, q975, q025 = compute_mean_estimations(X_train)

            for i, term in enumerate(X_train.columns):
                data = pd.DataFrame()
                
                # Plot theoretical f(x)
                data['x'] = X_train[str(i)]
                data['y'] = fs_train[str(i)]
                sns.lineplot(data = data, x='x', y='y', color='red', ax=axs[i])

                #Plot mean estimation
                data = pd.DataFrame()
                data['x'] = X_test[str(i)]
                data['y'] = mean_estimations[i]
                sns.lineplot(data = data, x='x', y='y', ax=axs[i], 
                            color='green', linestyle='--', alpha=0.7)
                
                # calculate confidence interval at 95%
                """
                ci = 1.96 * np.std(data['y'])/np.sqrt(len(data['x']))
                
                data['y+ci'] = data['y'] + ci
                data['y-ci'] = data['y'] - ci
                sns.lineplot(data = data, x='x', y='y+ci', color='grey', linestyle='--', alpha=0.5, ax=axs[i])
                sns.lineplot(data = data, x='x', y='y-ci', color='grey', linestyle='--', alpha=0.5, ax=axs[i])
                """
                # draw quantiles
                data['q975'] = q975[i]
                data['q025'] = q025[i]
                sns.lineplot(data = data, x='x', y='q975', color='grey', linestyle='--', alpha=0.5, ax=axs[i])
                sns.lineplot(data = data, x='x', y='q025', color='grey', linestyle='--', alpha=0.5, ax=axs[i])
                
            #plt.subplots_adjust(right=0.85) #make space for the legend
            
            axs[0].set_title("f(x) = x\N{SUBSCRIPT ONE}\u00b2")
            axs[1].set_title("f(x) = 2x\N{SUBSCRIPT TWO}")
            axs[2].set_title("f(x) = sen(x\N{SUBSCRIPT THREE})")
            
            theoretical_patch = mpatches.Patch(color='red', label='Theoretical f(x)')
            learned_patch = mpatches.Patch(color='green', label='Mean estimation of f(x)')
            quantiles = mpatches.Patch(color='grey', label='2.5% and 97.5% simulation quantiles')
            
            fig.legend(handles=[theoretical_patch, learned_patch, quantiles], loc='lower center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
            plt.tight_layout()
            #plt.show(block=True)
            plt.savefig("./" + type + ".png", dpi = 300, bbox_inches = "tight")

        except Exception as e:
            print(e) 
            import traceback
            traceback.print_exc()
            print("error on it type={0}".format(type))
            print(range)
