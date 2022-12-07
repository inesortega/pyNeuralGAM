import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import models as nam_models
import graph_builder
import os.path as osp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    default="./dataset/google",
    dest="input",
    type=str,
    help="""Input folder - place here X_train, y_train, etc..."""
)
parser.add_argument(
    "-o",
    "--output",
    default="results",
    dest="output",
    type=str,
    help="""Output folder - results and model"""
)
parser.add_argument(
    "-f",
    "--family",
    default="gaussian",
    type=str,
    metavar="Distribution Family. Use gaussian for LINEAR REGRESSION and binomial for LOGISTIC REGRESSION"
)
parser.add_argument(
    "-t",
    "--iteration",
    default=1,
    dest="iteration",
    type=int,
    help="""Iteration number"""
)

def partition(lst, batch_size):
    lst_len = len(lst)
    index = 0
    while index < lst_len:
      x = lst[index: batch_size + index]
      index += batch_size
      yield x

def generate_predictions(gen, nn_model):
  y_pred = []
  while True:
    try:
      x = next(gen)
      pred = nn_model(x).numpy()
      y_pred.extend(pred)
    except:
      break
  return y_pred

def get_test_predictions(nn_model, x_test, batch_size=256):
  batch_size = min(batch_size, x_test.shape[0])
  generator = partition(x_test, batch_size)
  return generate_predictions(generator, nn_model)

def get_feature_predictions(nn_model, features, batch_size=256):
  """Get feature predictions for unique values for each feature."""
  unique_feature_pred, unique_feature_gen = [], []
  for i, feature in enumerate(features):
    batch_size = min(batch_size, feature.shape[0])
    generator = partition(feature, batch_size)
    feature_pred = lambda x: nn_model.feature_nns[i](
        x, training=nn_model._false)  # pylint: disable=protected-access
    unique_feature_gen.append(generator)
    unique_feature_pred.append(feature_pred)

  feature_predictions = [
      generate_predictions(generator, feature_pred) for
      feature_pred, generator in zip(unique_feature_pred, unique_feature_gen)
  ]
  feature_predictions = [np.array(x) for x in feature_predictions]
  return feature_predictions

def compute_features(x_data):
  single_features = np.split(x_data, x_data.shape[1], axis=1)
  #unique_features = [np.unique(f, axis=0) for f in single_features]
  return single_features

def compute_mean_feature_importance(avg_hist_data, MEAN_PRED):
  mean_abs_score = {}
  for k in avg_hist_data:
    mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - MEAN_PRED[k]))
  x1, x2 = zip(*mean_abs_score.items())
  return x1, x2

def plot_mean_feature_importance(x1, x2, cols, width = 0.3):
  fig = plt.figure(figsize=(5, 4))
  ind = np.arange(len(x1))  # the x locations for the groups
  x1_indices = np.argsort(x2)
  cols_here = [cols[i] for i in x1_indices]
  # x1_here = [x12[i] for i in x1_indices]
  x2_here = [x2[i] for i in x1_indices]

  plt.bar(ind, x2_here, width, label='NAMs')
  # plt.bar(ind+width, x1_here, width, label='EBMs')
  plt.xticks(ind + width/2, cols_here, rotation=90, fontsize='large')
  plt.ylabel('Mean Absolute Score', fontsize='x-large')
  plt.legend(loc='top right', fontsize='large')
  plt.title(f'Overall Importance', fontsize='x-large')
  plt.show()
  return fig

def inverse_min_max_scaler(x, min_val, max_val):
  return (x + 1)/2 * (max_val - min_val) + min_val

def compute_mean_feature_importance(avg_hist_data):
  mean_abs_score = {}
  for k in avg_hist_data:
    mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - MEAN_PRED[k]))
  x1, x2 = zip(*mean_abs_score.items())
  return x1, x2

def plot_mean_feature_importance(x1, x2, width = 0.3):
  fig = plt.figure(figsize=(5, 4))
  ind = np.arange(len(x1))  # the x locations for the groups
  x1_indices = np.argsort(x2)
  cols_here = [cols[i] for i in x1_indices]
  # x1_here = [x12[i] for i in x1_indices]
  x2_here = [x2[i] for i in x1_indices]

  plt.bar(ind, x2_here, width, label='NAMs')
  # plt.bar(ind+width, x1_here, width, label='EBMs')
  plt.xticks(ind + width/2, cols_here, rotation=90, fontsize='large')
  plt.ylabel('Mean Absolute Score', fontsize='x-large')
  plt.legend(loc='top right', fontsize='large')
  plt.title(f'Overall Importance:', fontsize='x-large')
  plt.show()
  return fig

def shade_by_density_blocks(hist_data, num_rows, num_cols, 
                            UNIQUE_FEATURES_ORIGINAL,
                            SINGLE_FEATURES_ORIGINAL,
                            n_blocks=5, color=[0.9, 0.5, 0.5], 
                            feature_to_use=None, ):
  import matplotlib.patches as patches

  hist_data_pairs = list(hist_data.items())
  hist_data_pairs.sort(key=lambda x: x[0])
  min_y = np.min([np.min(a[1]) for a in hist_data_pairs])
  max_y = np.max([np.max(a[1]) for a in hist_data_pairs])
  min_max_dif = max_y - min_y
  min_y = min_y - 0.01 * min_max_dif
  max_y = max_y + 0.01 * min_max_dif

  if feature_to_use:
    hist_data_pairs = [v for v in hist_data_pairs if v[0] in feature_to_use] 

  for i, (name, pred) in enumerate(hist_data_pairs):

    # unique_x_data, single_feature_data, pred = data
    unique_x_data = UNIQUE_FEATURES_ORIGINAL[name]
    single_feature_data = SINGLE_FEATURES_ORIGINAL[name]
    ax = plt.subplot(num_rows, num_cols, i+1)
    min_x = np.min(unique_x_data)
    max_x = np.max(unique_x_data)
    x_n_blocks = min(n_blocks, len(unique_x_data))
    segments = (max_x - min_x) / x_n_blocks
    density = np.histogram(single_feature_data, bins=x_n_blocks)
    normed_density = density[0] / np.max(density[0])
    rect_params = []
    for p in range(x_n_blocks):
      start_x = min_x + segments * p
      end_x = min_x + segments * (p + 1)
      # start_insert_index = min(
      #     max(0, np.searchsorted(density[1], start_x) - 1), x_n_blocks - 1)
      # end_insert_index = min(
      #     max(0, np.searchsorted(density[1], end_x) - 1), x_n_blocks - 1)
      # d = (normed_density[start_insert_index] + normed_density[end_insert_index])/2
      d = min(1.0, 0.01 + normed_density[p])
      rect_params.append((d, start_x, end_x))

    for param in rect_params:
      alpha, start_x, end_x = param 
      rect = patches.Rectangle((start_x, min_y - 1), end_x - start_x, 
                               max_y - min_y + 1, linewidth=0.01, 
                              edgecolor=color, facecolor=color, alpha=alpha)
      ax.add_patch(rect)

def plot_all_hist(hist_data, num_rows, num_cols,  color_base, labels, MEAN_PRED, UNIQUE_FEATURES_ORIGINAL,
                  linewidth=3.0, min_y=None, max_y=None, alpha=1.0):
  hist_data_pairs = list(hist_data.items())
  hist_data_pairs.sort(key=lambda x: x[0])
  if min_y is None:
    min_y = np.min([np.min(a) for _, a in hist_data_pairs])
  if max_y is None:
    max_y = np.max([np.max(a) for _, a in hist_data_pairs])
  min_max_dif = max_y - min_y
  min_y = min_y - 0.01 * min_max_dif
  max_y = max_y + 0.01 * min_max_dif
  total_mean_bias = 0

  
  for i, (name, pred) in enumerate(hist_data_pairs):
    mean_pred = MEAN_PRED[name] #np.mean(pred)
    total_mean_bias += mean_pred
    unique_x_data = UNIQUE_FEATURES_ORIGINAL[name]
    plt.subplot(num_rows, num_cols, i+1)

    plt.plot(unique_x_data, pred - mean_pred, color=color_base, 
               linewidth=linewidth, alpha=alpha)
    plt.xticks(fontsize='x-large')
    
    plt.ylim(min_y, max_y)
    plt.yticks(fontsize='x-large')
    min_x = np.min(unique_x_data)
    max_x = np.max(unique_x_data)
    plt.xlim(min_x, max_x)
    if i % num_cols == 0:
      plt.ylabel('House Price Contribution', fontsize='x-large')
  return min_y, max_y

if __name__ == "__main__":
  import pandas as pd
  import os.path as osp
  import os

  args = parser.parse_args()
  variables = vars(args)
  
  family = variables["family"]    # gaussian / binomial
  iteration = variables["iteration"]
  
  output_folder = os.path.normpath(os.path.abspath(os.path.join("./", variables["output"])))

  if not os.path.exists(output_folder):
      os.makedirs(output_folder)

  input_path = os.path.normpath(os.path.abspath(os.path.join("./", variables["input"])))
  
  print("Starting --- INPUT {0}, OUTPUT {1}".format(input_path, output_folder))

  regression = False
  if family == "gaussian":
    regression = True

  variables.pop("iteration")
  print(variables)
  
  if iteration is not None:
    path = os.path.normpath(os.path.join(output_folder, str(iteration)))
    #add iteration
    if not os.path.exists(path):
        os.makedirs(path)

  else:
    path = os.path.normpath(output_folder)
      
  column_names = ["f1", "f2", "f3"]

  X_train = pd.read_csv("{0}/X_train.csv".format(input_path), index_col=0, dtype=np.float32).reset_index(drop=True)
  X_train.columns = X_train.columns.map(int)
  X_train = X_train.to_numpy()
  
  fs_train = pd.read_csv("{0}/fs_train.csv".format(input_path), index_col=0).reset_index(drop=True)
  
  y_train = pd.read_csv("{0}/y_train.csv".format(input_path), index_col=0, dtype=np.float32).reset_index(drop=True).squeeze()
  y_train = y_train.to_numpy()

  X_test = pd.read_csv("{0}/X_test.csv".format(input_path), index_col=0, dtype=np.float32).reset_index(drop=True)
  X_test = X_test.to_numpy()

  fs_test = pd.read_csv("{0}/fs_test.csv".format(input_path), index_col=0).reset_index(drop=True)
  y_test = pd.read_csv("{0}/y_test.csv".format(input_path), index_col=0, dtype=np.float32).reset_index(drop=True).squeeze()
  y_test = y_test.to_numpy()

  tf.compat.v1.reset_default_graph()
  nn_model = graph_builder.create_nam_model(
            x_train=X_train,
            dropout=0.0,
            feature_dropout=0.0,
            activation="exu",
            num_basis_functions=[1024,512,256],
            shallow=False,
            units_multiplier=2,
            trainable=False,
            name_scope="model_0")

  _ = nn_model(X_train[:1])
  nn_model.summary()
  
  logdir = path + "/model"
  ckpt_dir = osp.normpath(osp.join(logdir, 'model_0', 'best_checkpoint'))

  print(ckpt_dir)
  ckpt_files = sorted(tf.io.gfile.listdir(ckpt_dir))
  ckpt = osp.join(ckpt_dir, ckpt_files[0].split('.data')[0])
  ckpt_reader = tf.train.load_checkpoint(ckpt)
  ckpt_vars = tf.train.list_variables(ckpt)

  """for var in ckpt_vars:
        tensor_name = var[0]
        value = ckpt_reader.get_tensor(tensor_name)
        # Find equivalent tensor in nn_model
        i_elem = [i for i,x in enumerate(nn_model.variables) if tensor_name.split(":")[-1] == x.name]
        if len(i_elem) == 1:
          nn_model.variables[i_elem[0]].assign(value)
  #manually assign bias! 
  nam_bias = ckpt_reader.get_tensor("model_0/nam/bias")
  nn_model.variables[-1].assign(nam_bias)"""

  for var in nn_model.variables:
    tensor_name = var.name.split(':', 1)[0].replace('nam', 'model_0/nam')
    value = ckpt_reader.get_tensor(tensor_name)
    var.assign(value)

  test_predictions = get_test_predictions(nn_model, X_test)
  train_predictions = get_test_predictions(nn_model, X_train)
  unique_features = compute_features(X_test)
  unique_features_training = compute_features(X_train)

  feature_predictions = get_feature_predictions(nn_model, unique_features)
  feature_predictions_training = get_feature_predictions(nn_model, unique_features_training)

  feature_predictions = {}
  for i, feat in enumerate(unique_features):
    feature_predictions[column_names[i]] = nn_model.feature_nns[i](feat, training=nn_model._false)

  feature_predictions_training = {}
  for i, feat in enumerate(unique_features_training):
    feature_predictions_training[column_names[i]] = nn_model.feature_nns[i](feat, training=nn_model._false)

  if regression is False:
    # compute both ROC-AUC
    y_test_binomial = pd.read_csv(path + "/y_test_binomial.csv", index_col=0, dtype=np.float32).reset_index(drop=True).squeeze()
    y_test_binomial = y_test_binomial.to_numpy()

    y_train_binomial = pd.read_csv(path + "/y_train_binomial.csv", index_col=0, dtype=np.float32).reset_index(drop=True).squeeze()
    y_train_binomial = y_train_binomial.to_numpy()

    test_metric = graph_builder.calculate_metric(y_test_binomial, test_predictions, regression=False)
    train_metric = graph_builder.calculate_metric(y_train_binomial, train_predictions, regression=False)
    variables["auc_roc"] = test_metric


    def sigmoid(x):
      """Sigmoid function."""
      if isinstance(x, list):
        x = np.array(x)
      return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    test_metric = graph_builder.calculate_metric(y_test, sigmoid(test_predictions), regression=True)
    train_metric = graph_builder.calculate_metric(y_train, sigmoid(train_predictions), regression=True)

  else:
    # For gaussian case, compute as regression=True
    test_metric = graph_builder.calculate_metric(y_test, test_predictions, regression=True)
    train_metric = graph_builder.calculate_metric(y_train, train_predictions, regression=True)

  variables["err_test_rmse"] = test_metric
  variables["err_rmse"] = train_metric

  variables["err_test"] = test_metric**2
  variables["err"] = train_metric**2

  NUM_FEATURES = X_test.shape[1]
  SINGLE_FEATURES = np.split(X_test, NUM_FEATURES, axis=1)
  UNIQUE_FEATURES = SINGLE_FEATURES
  #UNIQUE_FEATURES = [np.unique(x, axis=0) for x in SINGLE_FEATURES]
  
  SINGLE_FEATURES_ORIGINAL = {}
  UNIQUE_FEATURES_ORIGINAL = {}

  X_test = pd.DataFrame(X_test, columns=column_names)

  col_min_max = {}
  for col in X_test.columns:
    unique_vals = X_test[col].unique()
    col_min_max[col] = (np.min(unique_vals), np.max(unique_vals))

  for i, col in enumerate(column_names):
    min_val, max_val = col_min_max[col]
    UNIQUE_FEATURES_ORIGINAL[col] = inverse_min_max_scaler(
        UNIQUE_FEATURES[i][:, 0], min_val, max_val)
    SINGLE_FEATURES_ORIGINAL[col] = inverse_min_max_scaler(
        SINGLE_FEATURES[i][:, 0], min_val, max_val)

  fs_pred = pd.DataFrame.from_dict(feature_predictions)
  fs_train = pd.DataFrame.from_dict(feature_predictions_training)
  x_pred = pd.DataFrame(UNIQUE_FEATURES_ORIGINAL)

  x_list = [X_test, x_pred]
  fs_list = [fs_test, fs_pred]

  """ SAVE RESULTS"""
  pd.DataFrame(fs_pred).to_csv(path + "/fs_test_estimated.csv")
  pd.DataFrame(fs_train).to_csv(path + "/fs_train_estimated.csv")  

  pd.DataFrame(test_predictions).to_csv(path + "/y_pred.csv")
  pd.DataFrame.from_dict(variables, orient="index").transpose().to_csv(path + "/variables.csv", index=False)