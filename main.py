import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from src.utils.utils import generate_data, plot_confusion_matrix, plot_multiple_partial_dependencies, plot_predicted_vs_real, plot_partial_dependencies, plot_y_histogram, split
from src.NeuralGAM.ngam import NeuralGAM, load_model
import mlflow
from sklearn.metrics import mean_squared_error
        
parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help='Choose wether to use Linear (linear) or Logistic (logistic) Regression')

linear_regression_parser = subparsers.add_parser(name='linear', help="Linear Regression")
linear_regression_parser.add_argument(
    "-t",
    "--type",
    default="homoscedastic",
    metavar="{homoscedastic, heteroscedastic} ",
    dest="type",
    type=str,
    help="""Choose wether to generate a homoscesdastic or heteroscedastic epsilon term"""
)
linear_regression_parser.add_argument(
    "-d",
    "--distribution",
    default="uniform",
    type=str,
    metavar="{uniform, normal} ",
    help="Choose wether to generate normal or uniform distributed dataset"
)
linear_regression_parser.set_defaults(link='linear')

logistic_regression_parser = subparsers.add_parser(name='logistic', help="Logistic Regression")
logistic_regression_parser.add_argument(
    "-d",
    "--distribution",
    default="uniform",
    type=str,
    metavar="{uniform, normal} ",
    help="Choose wether to generate normal or uniform distributed dataset"
)
logistic_regression_parser.set_defaults(link='logistic')

def setup_mlflow():
    
    MLFLOW_URI = "http://10.11.1.21:5000/"
    MLFLOW_EXP = "iortega.neuralGAM"
    # Setting the MLflow tracking server
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # Setting the requried environment variables
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9001'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio_user'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio_root_pass'

    exp = mlflow.get_experiment_by_name(MLFLOW_EXP)
    if not exp:
        mlflow.create_experiment(MLFLOW_EXP)
    else:
        if exp.lifecycle_stage == 'deleted':
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.restore_experiment(exp.experiment_id)
            print('{} deleted experiment recovered and ready to use'.format(MLFLOW_EXP))

    mlflow.set_experiment(MLFLOW_EXP)

if __name__ == "__main__":
    
    """ 
    USAGE 
    
    LINEAR REGRESSION: 
        python main.py linear -t {homoscedastic, heteroscedastic} -d {uniform, normal}
    LOGISTIC REGRESSION: 
        python main.py logistic -d {uniform, normal}
    """
    
    args = parser.parse_args()
    variables = vars(args)
    
    type = variables.get("type", None)
    distribution = variables["distribution"]
    link = variables["link"]
    
    print(variables)
    
    path = os.path.normpath(os.path.abspath("./results_test/{0}_{1}_{2}".format(type, distribution, link)))
    if not os.path.exists(path):
        os.mkdir(path)
        
    X, y, fs = generate_data(nrows=25000, type=type, distribution=distribution, link=link, output_folder=path)
    
    print("X")
    print(X.describe())
    print("\ny")
    print(y)
    
    print("\n Number of elements per class on training set")
    print(pd.DataFrame(np.where(y >= 0.5, 1, 0)).value_counts())
    X_train, X_test, y_train, y_test = split(X, y)
    
    with mlflow.start_run():
        
        if not __debug__ and os.path.isfile(path + "/model.ngam"):
            ngam = load_model(path + "/model.ngam")
        else:
            ngam = NeuralGAM(num_inputs = len(X_train.columns), link=link)
            
            if link == "logistic":
                y_train_binomial = np.random.binomial(n=1, p=y_train, size=y_train.shape[0])
                ngam.fit(X_train = X_train, y_train = y_train_binomial, max_iter = 5, convergence_threshold=0.04)
            else:         
                ngam.fit(X_train = X_train, y_train = y_train, max_iter = 5, convergence_threshold=0.04)

            training_mse = ngam.training_mse
            model_path = ngam.save_model(path)
            
            #mlflow.log_artifact(model_path)
            
            print("Achieved RMSE during training = {0}".format(mean_squared_error(y_train, ngam.y, squared=False)))
            
        y_pred = ngam.predict(X_test)
        
        #Calculo el error entre y_pred (prob teorica) y el resultado
        mse = mean_squared_error(y_test, y_pred)
        
        training_fs = ngam.get_partial_dependencies(X_train)
        test_fs = ngam.get_partial_dependencies(X_test)
        
        print("Finished predictions...Plotting results... ")
        print(variables)
        if link == "logistic":
            x_list = [X_train, X_test]
            fs_list = [training_fs, test_fs]
            legends = ["X_train", "X_test"]
            
            variables["MSE"] = mse            
            plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title=variables, output_path=path + "/functions_logistic.png")
            
            legends = ["y_train", "y_cal", "y_test", "y_pred"]
            plot_y_histogram([y_train, ngam.y, y_test, y_pred], legends=legends, title=variables, output_path=path + "/y_histogram.png")

            y_test_bin = np.where(y_test >= 0.5, 1, 0)
            y_pred_bin = np.where(y_test >= 0.5, 1, 0)
            
            print(classification_report(y_true=y_test_bin, y_pred=y_pred_bin))
            tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin).ravel()
            cm_normalized = confusion_matrix(y_test_bin, y_pred_bin, normalize="true")
            
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
            
        mse = mean_squared_error(y_test, y_pred)
        
        x_list = [X, X_train, X_test]
        fs_list = [fs, training_fs, test_fs]
        legends = ["X", "X_train", "X_test"]
        plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title="MSE = {0}".format(mse), output_path=path + "/functions.png")
        
    plt.show(block=True)