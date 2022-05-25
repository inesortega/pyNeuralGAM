import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, log_loss
from src.utils.utils import generate_data, plot_confusion_matrix, plot_multiple_partial_dependencies, plot_predicted_vs_real, plot_partial_dependencies, plot_y_histogram, split
from src.NeuralGAM.ngam import NeuralGAM, load_model, compute_loss
import mlflow
from sklearn.metrics import mean_squared_error
import pandas as pd
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
linear_regression_parser.add_argument(
    "-i",
    "--iteration",
    default=None,
    type=int,
    metavar="N_iteration (for simulations)"
)
linear_regression_parser.set_defaults(family='gaussian')

logistic_regression_parser = subparsers.add_parser(name='logistic', help="Logistic Regression")
logistic_regression_parser.add_argument(
    "-d",
    "--distribution",
    default="uniform",
    type=str,
    metavar="{uniform, normal} ",
    help="Choose wether to generate normal or uniform distributed dataset"
)
logistic_regression_parser.set_defaults(family='binomial')

def setup_mlflow(family = None):
    
    print("Setting up MLFLOW....")

    MLFLOW_URI = "http://10.11.1.21:5000/"
    MLFLOW_EXP = "simulation.neuralGAM"
    if family:
        MLFLOW_EXP = MLFLOW_EXP + "." + family
    # Setting the MLflow tracking server
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # Set the requried environment variables on .env
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
    print("Tracking run on exp " + MLFLOW_EXP)

if __name__ == "__main__":
    
    """ 
    LINEAR REGRESSION: 
        python main.py linear -t {homoscedastic, heteroscedastic} -d {uniform, normal}
    LOGISTIC REGRESSION: 
        python main.py logistic -d {uniform, normal}
    """
    
    args = parser.parse_args()
    variables = vars(args)
    
    type = variables.get("type", None)
    distribution = variables["distribution"]
    family = variables["family"]    # gaussian / binomial
    iteration = variables["iteration"]
    variables.pop("iteration")
    print(variables)
    
    setup_mlflow(family)
    
    if iteration is not None:
        rel_path = "./results/{0}".format(iteration)
        path = os.path.normpath(os.path.abspath(rel_path))
        #add iteration
        if not os.path.exists(path):
            os.mkdir(path)

    else:
        rel_path = "./results/"
        path = os.path.normpath(os.path.abspath(rel_path))
        
    # add exec type
    path = path + "/" + "_".join(list(variables.values())) 
    if not os.path.exists(path):
        os.mkdir(path)

    X, y, fs = generate_data(nrows=25000, type=type, distribution=distribution, family=family, output_folder=path)
    
    print("\n Number of elements per class on training set")
    print(pd.DataFrame(np.where(y >= 0.5, 1, 0)).value_counts())
    X_train, X_test, y_train, y_test, fs_train, fs_test = split(X, y, fs)
    
    with mlflow.start_run():
        
        if iteration is not None:
            mlflow.log_param("it", iteration)
        ngam = NeuralGAM(num_inputs = len(X_train.columns), family=family)

        conv_threshold = 0.00001
        delta_threshold = 0.01
        variables["convergence_threshold"] = conv_threshold
        variables["delta_threshold"] = delta_threshold
        mlflow.log_params(dict(variables))
            
        if family == "binomial":
            y_train_binomial = np.random.binomial(n=1, p=y_train, size=y_train.shape[0])
            muhat, gs_train = ngam.fit(X_train = X_train, y_train = y_train_binomial, max_iter = 10, convergence_threshold=conv_threshold, delta_threshold=delta_threshold)
        else:
            muhat, gs_train = ngam.fit(X_train = X_train, y_train = y_train, max_iter = 10, convergence_threshold=conv_threshold, delta_threshold=delta_threshold)
                        
        #model_path = ngam.save_model(path)
        #mlflow.pyfunc.log_model(python_model=ngam, artifact_path='neuralGAM')

        mlflow.log_param("eta0", ngam.eta0)

        y_pred = ngam.predict(X_test)
        fs_pred = ngam.get_partial_dependencies(X_test)
        
        mse = mean_squared_error(y_train, ngam.y)
        pred_mse = mean_squared_error(y_test, y_pred)
        
        mlflow.log_metric("y_train_mse", mse)
        mlflow.log_metric("y_pred_mse", pred_mse)
        print("MSE y_train / ngam.y = {0}".format(mse))
        print("MSE y_test / y_pred = {0}".format(pred_mse))
        
        print("Finished predictions...Plotting results... ")

        x_list = [X_train, X_train]
        fs_list = [fs_train, gs_train]
        legends = ["real", "estimated_training"]
        vars = variables
        vars["err"] = round(mse, 4)
        plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title=vars, output_path=path + "/fs_training.png")
        x_list = [X_test, X_test]
        fs_list = [fs_test, fs_pred]
        legends = ["real_test", "estimated_test"]
        vars = variables
        vars["err"] = round(pred_mse, 4)
        plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title=vars, output_path=path + "/fs_test.png")
        
        pd.DataFrame(X_train).to_csv(path + "/X_train.csv")
        pd.DataFrame(y_train).to_csv(path + "/y_train.csv")
        pd.DataFrame(fs_train).to_csv(path + "/fs_train.csv")
        pd.DataFrame(gs_train).to_csv(path + "/fs_train_estimated.csv")

        pd.DataFrame(X_test).to_csv(path + "/X_test.csv")
        pd.DataFrame(y_test).to_csv(path + "/y_test.csv")
        pd.DataFrame(fs_test).to_csv(path + "/fs_test.csv")
        pd.DataFrame(fs_pred).to_csv(path + "/fs_test_estimated.csv")
        pd.DataFrame(y_pred).to_csv(path + "/y_pred.csv")

        mlflow.log_artifacts(path)
        mlflow.end_run("FINISHED")
           
    #plt.show(block=True)
    