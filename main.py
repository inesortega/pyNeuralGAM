import argparse
import os
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from src.utils.utils import generate_data, plot_confusion_matrix, plot_multiple_partial_dependencies, plot_predicted_vs_real, plot_partial_dependencies, plot_y_histogram, split
from src.NeuralGAM.ngam import NeuralGAM, load_model

parser = argparse.ArgumentParser()

parser.add_argument(
    "-t",
    "--type",
    default="homoscedastic",
    metavar="{homoscedastic, heteroscedastic} ",
    dest="data_type",
    type=str,
    help="""Choose wether to generate a homoscesdastic or heteroscedastic dataset"""
)
parser.add_argument(
    "-d",
    "--distribution",
    default="uniform",
    type=str,
    metavar="{uniform, normal} ",
    help="Choose wether to generate normal or uniform distributed dataset"
)
parser.add_argument(
    "-l",
    "--link",
    default="identity",
    type=str,
    metavar="{identity, binomial} ",
    help="Choose wether the response Y is continuous or binomial (0/1)"
)


if __name__ == "__main__":
    
    """ 
    USAGE 
    
    Uniform distribution:
        python main.py -t homoscedastic -d uniform
        python main.py -t heteroscedastic -d uniform
    
    Normal Distribution:
        python main.py -t homoscedastic -d uniform
        python main.py -t heteroscedastic -d uniform
    """
    
    args = parser.parse_args()
    variables = vars(args)
    
    data_type = variables["data_type"]
    distribution = variables["distribution"]
    link = variables["link"]
    
    print(variables)
    
    path = os.path.normpath(os.path.abspath("./results/{0}_{1}_{2}".format(data_type, distribution, link)))
    if not os.path.exists(path):
        os.mkdir(path)
        
    X, y, fs = generate_data(nrows=25000, data_type=data_type, distribution=distribution, link=link, output_folder=path)
    
    print("X")
    print(X.describe())
    print("\ny")
    print(y.describe())
    
    X_train, X_test, y_train, y_test = split(X, y)
    
    if not __debug__ and os.path.isfile(path + "/model.ngam"):
        ngam = load_model(path + "/model.ngam")
    else:
        ngam = NeuralGAM(num_inputs = len(X_train.columns), link=link)
        ycal, mse = ngam.fit(X_train = X_train, y_train = y_train, max_iter = 10, convergence_threshold=0.04)
        ngam.save_model(path)
        print("Achieved RMSE during training = {0}".format(mean_squared_error(y_train, ngam.y, squared=False)))
        
    y_pred = ngam.predict(X_test)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    
    training_fs = ngam.get_partial_dependencies(X_train)
    test_fs = ngam.get_partial_dependencies(X_test)
    
    print("Finished predictions...Plotting results...")
    x_list = [X_train, X_test]
    fs_list = [training_fs, test_fs]
    legends = ["training", "test"]
    plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title="Learned and Predicted Functions", output_path=path + "/functions.png")
    
    if link == "binomial":
        plot_y_histogram([y, y_test, y_pred], ["original", "test", "predicted"], title="MSE on prediction = {0}".format(mse))
    
        #Compute classification metrics
        import numpy as np
        y_true_bin = np.where(y_test >= 0.5, 1, 0)
        y_pred_bin = np.where(y_pred >= 0.5, 1, 0)
        print(classification_report(y_true=y_true_bin, y_pred=y_pred_bin))
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        cm_normalized = confusion_matrix(y_true_bin, y_pred_bin, normalize="true")
        
        plot_confusion_matrix(cm_normalized, ['0', '1'],
                                    path + '/confusion-matrix.png',
                                    title='Confusion Matrix')
        metrics = dict()
        metrics["tp"] = tp
        metrics["fp"] = fp
        metrics["tn"] = tn
        metrics["fn"] = fn
        metrics["precision"] = tp / (tp + fp)
        metrics["recall"] = tp / (tp + fn)
        metrics["tnr"] = tn / (tn + fp)
        metrics["accuracy"] = (metrics["tp"] + metrics["tn"]) / (metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"])
        metrics["f1"] = 2 * ((metrics["precision"]  * metrics["recall"]) / (metrics["precision"]  + metrics["recall"]))
        
        import pandas as pd
        pd.DataFrame([metrics]).to_csv(path + '/classification-report.csv', index=False)

        
    plt.show(block=True)