import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, log_loss
from src.utils.utils import generate_data, plot_confusion_matrix, plot_multiple_partial_dependencies, plot_predicted_vs_real, plot_partial_dependencies, plot_y_histogram, split
from src.NeuralGAM.ngam import NeuralGAM, load_model, apply_link, compute_loss
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

def setup_mlflow():
    
    MLFLOW_URI = "http://10.11.1.21:5000/"
    MLFLOW_EXP = "iortega.neuralGAM"
    # Setting the MLflow tracking server
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # Setting the requried environment variables
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://10.11.1.21:9000'
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
    
    print(variables)
    
    #setup_mlflow()
    log_mlflow = False
    rel_path = "./results_test_v2/{0}".format("_".join(list(variables.values())))
    path = os.path.normpath(os.path.abspath(rel_path))
    if not os.path.exists(path):
        os.mkdir(path)
        
    X, y, fs = generate_data(nrows=25000, type=type, distribution=distribution, family=family, output_folder=path)
    
    print("X")
    print(X.describe())
    print("\ny")
    print(y)
    
    print("\n Number of elements per class on training set")
    print(pd.DataFrame(np.where(y >= 0.5, 1, 0)).value_counts())
    X_train, X_test, y_train, y_test = split(X, y)
    
    fs_train = fs.loc[X_train.index, :]
    fs_test = fs.loc[X_test.index, :]
    
    #with mlflow.start_run():
        
    if not __debug__ and os.path.isfile(path + "/model.ngam"):
        ngam = load_model(path + "/model.ngam")
    else:
        ngam = NeuralGAM(num_inputs = len(X_train.columns), family=family)
        
        if family == "binomial":
            y_train_binomial = np.random.binomial(n=1, p=y_train, size=y_train.shape[0])
            muhat, partial = ngam.fit(X_train = X_train, y_train = y_train_binomial, max_iter = 5, convergence_threshold=0.1)
            print("MSE y_train / ngam.y = {0}".format(mean_squared_error(y_train, ngam.y)))
                            
        elif family == "gaussian":         
            muhat, partial = ngam.fit(X_train = X_train, y_train = y_train, max_iter = 5, convergence_threshold=0.05)
            print("MSE y_train / ngam.y = {0}".format(mean_squared_error(y_train, ngam.y)))

        #muhat2 = partial.apply(lambda x: apply_link(family, x))

        x_list = [X_train, X_train]
        fs_list = [fs_train, partial]
        legends = ["real", "estimated"]
        plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title=variables, output_path=path + "/functions_logistic.png")
        
        
        #model_path = ngam.save_model(path)
        #[mlflow.log_metric("training_mse", float(i)) for i in ngam.training_mse]
        #mlflow.pyfunc.log_model(python_model=ngam, artifact_path='neuralGAM')
        
    #y_pred = ngam.predict(X_test)
    #fs_pred = ngam.get_partial_dependencies(X_test, xform=False)
    
    #Calculo el error entre y_pred (prob teorica) y el resultado    
            
    #print("Finished predictions...Plotting results... ")
    #print(variables)
    """
    if family == "binomial":
        
        training_fs = ngam.get_partial_dependencies(X_train, xform=False)
        test_fs = ngam.get_partial_dependencies(X_test, xform=False)
        x_list = [X_train, X_test]
        fs_list = [training_fs, test_fs]
        legends = ["X_train", "X_test"]
        
        plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title=variables, output_path=path + "/functions_logistic.png")
        
        legends = ["y_train", "y_cal", "y_test", "y_pred"]
        plot_y_histogram([y_train, ngam.y, y_test, y_pred], legends=legends, title=variables, output_path=path + "/y_histogram.png")
        
        p_success = ngam.muhat  # probability of success is the mean of y_train
        y_test_bin = np.where(y_test >= p_success, 1, 0)
        y_pred_bin = np.where(y_test >= p_success, 1, 0)
        
        err = compute_loss("binary_cross_entropy", y_test_bin, y_pred_bin)
        variables["err"] = err
        mlflow.log_params(dict(variables))
        
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
        mlflow.log_metrics(metrics)
        pd.DataFrame([metrics]).to_csv(path + '/classification-report.csv', index=False)      
    
    else:
        mse = mean_squared_error(y_test, y_pred)
        variables["err"] = mse
        mlflow.log_params(dict(variables))
        
    # Put all fs on the scale of the distribution mean by applying the inverse link function column-wise
    #fs = fs.apply(lambda x: apply_link(family, x))
    #training_fs = ngam.get_partial_dependencies(X_train, xform=True)
    #test_fs = ngam.get_partial_dependencies(X_test, xform=True)
    
    x_list = [X, X_train, X_test]
    fs_list = [fs, training_fs, test_fs]
    legends = ["X", "X_train", "X_test"]
    
    plot_multiple_partial_dependencies(x_list=x_list, f_list=fs_list, legends=legends, title=variables, output_path=path + "/functions.png")
    mlflow.log_artifacts(path)"""
        
plt.show(block=True)