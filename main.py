import argparse
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from src.utils.utils import generate_data, plot_predicted_vs_real, plot_partial_dependencies, split
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
    
    print(variables)
    
    path = os.path.normpath(os.path.abspath("./results/{0}_{1}_v2".format(data_type, distribution)))
    if not os.path.exists(path):
        os.mkdir(path)
        
    X, y = generate_data(nrows=25000, data_type=data_type, distribution=distribution, output_folder=path)
    print(X.describe())
    print(y.describe())
    
    X_train, X_test, y_train, y_test = split(X, y)
    
    if not __debug__ and os.path.isfile(path + "/model.ngam"):
        ngam = load_model(path + "/model.ngam")
    else:
        ngam = NeuralGAM(num_inputs = len(X_train.columns), num_units=100)
        ycal, mse = ngam.fit(X_train = X_train, y_train = y_train, max_iter = 100, convergence_threshold=0.1)
        ngam.save_model(path)
        print("Achieved RMSE during training = {0}".format(mean_squared_error(y_train, ngam.y, squared=False)))
        
    y_pred = ngam.predict(X_test)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    
    training_fs = ngam.get_partial_dependencies(X_train)
    test_fs = ngam.get_partial_dependencies(X_test)
    
    print("Finished predictions...Plotting results...")
    plot_predicted_vs_real([y_test, y_pred], ["real", "predicted"], title="MSE on prediction = {0}".format(mse), output_path=path + "/y_pred_vs_real.png")
    plot_partial_dependencies(X_test, test_fs, title="Prediction f(x) - {0} - {1}".format(data_type, distribution), output_path=path + "/pred_f.png")
    plot_partial_dependencies(X_train, training_fs, title="Learned f(x) from training {0} - {1}".format(data_type, distribution), output_path=path + "/training_f.png")
    plt.show(block=True)