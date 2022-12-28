import argparse
import os

parser = argparse.ArgumentParser()


parser.add_argument(
    "-o",
    "--output",
    default="dataset",
    dest="output",
    type=str,
    help="""Output folder - results and model"""
)

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

if __name__ == "__main__":
    
    """ 
    LINEAR REGRESSION: 
        python generate_dataset.py linear -t {homoscedastic, heteroscedastic} -d {uniform, normal}
    LOGISTIC REGRESSION: 
        python main.py logistic -d {uniform, normal}
    """
    
    args = parser.parse_args()
    variables = vars(args)
    type = variables.get("type", None)
    distribution = variables["distribution"]
    family = variables["family"]    # gaussian / binomial
    print(variables)
    # add exec type
    output = variables.pop("output")
    output_path = os.path.normpath(os.path.abspath(os.path.join("./", output)))
    data_type_path = "_".join(list(variables.values())) 
    output_path = os.path.join(output_path, data_type_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(output_path)
    from src.utils.utils import generate_data, split
    import pandas as pd
    import numpy as np

    np.random.seed(1231241351)

    type = variables.get("type", None)
    distribution = variables["distribution"]
    family = variables["family"]    # gaussian / binomial
    X, y, fs = generate_data(nrows=15000, type=type, distribution=distribution, family=family, output_folder=output_path)
    X_train, X_test, y_train, y_test, fs_train, fs_test = split(X, y, fs, test_size=0.2)
    
    pd.DataFrame(X_train).to_csv(output_path + "/X_train.csv")
    pd.DataFrame(y_train).to_csv(output_path + "/y_train.csv")
    pd.DataFrame(fs_train).to_csv(output_path + "/fs_train.csv")
    pd.DataFrame(X_test).to_csv(output_path + "/X_test.csv")
    pd.DataFrame(y_test).to_csv(output_path + "/y_test.csv")
    pd.DataFrame(fs_test).to_csv(output_path + "/fs_test.csv")