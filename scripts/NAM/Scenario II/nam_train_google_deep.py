# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Training script for Neural Additive Models.

"""
import operator
import os
from typing import Tuple, Iterator, List, Dict
import numpy as np
import tensorflow.compat.v1 as tf1

import data_utils
import graph_builder
import argparse

training_epochs = 1000
learning_rate = 0.00674
output_regularization =  0.0
l2_regularization = 1e-6
batch_size = 1024
decay_rate = 0.995
dropout = 0.0
feature_dropout = 0.0
num_basis_functions = [1024,512,256]
shallow = False
units_multiplier = 1
cross_val = False
max_checkpoints_to_keep = 1
save_checkpoint_every_n_epochs = 10
n_models = 1
fold_num = 1
activation = "exu"
debug = True
use_dnn = False
early_stopping_epochs = 100
_N_FOLDS = 1
gfile = tf1.io.gfile
DatasetType = data_utils.DatasetType

GraphOpsAndTensors = graph_builder.GraphOpsAndTensors
EvaluationMetric = graph_builder.EvaluationMetric

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

linear_regression_parser.add_argument(
    "-o",
    "--output",
    default="results",
    dest="output",
    type=str,
    help="""Output folder"""
)

"""linear_regression_parser.add_argument(
    "-n",
    "--neurons",
    default=1024,
    type=int,
    metavar="Number of neurons per hidden layer"
)

linear_regression_parser.add_argument(
    "-l",
    "--layers",
    default=1,
    type=int,
    metavar="Number of hidden layers"
)"""
linear_regression_parser.add_argument(
    "-c",
    "--convergence_threshold",
    default=0.00001,
    type=float,
    metavar="Convergence Threshold of Backfitting algorithm"
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

logistic_regression_parser.add_argument(
    "-i",
    "--iteration",
    default=None,
    type=int,
    metavar="N_iteration (for simulations)"
)

logistic_regression_parser.add_argument(
    "-c",
    "--convergence_threshold",
    default=0.00001,
    type=float,
    metavar="Convergence Threshold of backfitting algorithm"
)

logistic_regression_parser.add_argument(
    "-a",
    "--delta_threshold",
    default=0.01,
    type=float,
    metavar="Convergence Threshold of LS algorithm"
)
logistic_regression_parser.add_argument(
    "-o",
    "--output",
    default="results",
    dest="output",
    type=str,
    help="""Output folder"""
)

logistic_regression_parser.set_defaults(family='binomial')

def _get_train_and_lr_decay_ops(
    graph_tensors_and_ops,
    early_stopping):
  """Returns training and learning rate decay ops."""
  train_ops = [
      g['train_op']
      for n, g in enumerate(graph_tensors_and_ops)
      if not early_stopping[n]
  ]
  lr_decay_ops = [
      g['lr_decay_op']
      for n, g in enumerate(graph_tensors_and_ops)
      if not early_stopping[n]
  ]
  return train_ops, lr_decay_ops


def _update_latest_checkpoint(checkpoint_dir,
                              best_checkpoint_dir):
  """Updates the latest checkpoint in `best_checkpoint_dir` from `checkpoint_dir`."""
  for filename in gfile.glob(os.path.join(best_checkpoint_dir, 'model.*')):
    gfile.remove(filename)
  for name in gfile.glob(os.path.join(checkpoint_dir, 'model.*')):
    gfile.copy(
        name,
        os.path.join(best_checkpoint_dir, os.path.basename(name)),
        overwrite=True)


def _create_computation_graph(
    x_train, y_train, x_validation,
    y_validation, batch_size, regression
):
  """Build the computation graph."""
  graph_tensors_and_ops = []
  metric_scores = []
  for n in range(n_models):
    graph_tensors_and_ops_n, metric_scores_n = graph_builder.build_graph(
        x_train=x_train,
        y_train=y_train,
        x_test=x_validation,
        y_test=y_validation,
        activation=activation,
        learning_rate=learning_rate,
        batch_size=batch_size,
        shallow=shallow,
        output_regularization=output_regularization,
        l2_regularization=l2_regularization,
        dropout=dropout,
        num_basis_functions=num_basis_functions,
        units_multiplier=units_multiplier,
        decay_rate=decay_rate,
        feature_dropout=feature_dropout,
        regression=regression,
        use_dnn=use_dnn,
        trainable=True,
        name_scope=f'model_{n}')
    graph_tensors_and_ops.append(graph_tensors_and_ops_n)
    metric_scores.append(metric_scores_n)
  return graph_tensors_and_ops, metric_scores


def _create_graph_saver(graph_tensors_and_ops,
                        logdir, num_steps_per_epoch):
  """Create saving hook(s) as well as model and checkpoint directories."""
  saver_hooks, model_dirs, best_checkpoint_dirs = [], [], []
  save_steps = num_steps_per_epoch * save_checkpoint_every_n_epochs
  # The MonitoredTraining Session counter increments by `n_models`
  save_steps = save_steps * n_models
  for n in range(n_models):
    scaffold = tf1.train.Scaffold(
        saver=tf1.train.Saver(
            var_list=graph_tensors_and_ops[n]['nn_model'].trainable_variables,
            save_relative_paths=True,
            max_to_keep=max_checkpoints_to_keep))
    model_dirs.append(os.path.join(logdir, 'model_{}').format(n))
    best_checkpoint_dirs.append(os.path.join(model_dirs[-1], 'best_checkpoint'))
    gfile.makedirs(best_checkpoint_dirs[-1])
    saver_hook = tf1.train.CheckpointSaverHook(
        checkpoint_dir=model_dirs[-1], save_steps=save_steps, scaffold=scaffold)
    saver_hooks.append(saver_hook)
  return saver_hooks, model_dirs, best_checkpoint_dirs


def _update_metrics_and_checkpoints(sess,
                                    epoch,
                                    metric_scores,
                                    curr_best_epoch,
                                    best_validation_metric,
                                    best_train_metric,
                                    model_dir,
                                    best_checkpoint_dir,
                                    regression,
                                    metric_name = 'MSE'):
  """Update metric scores and latest checkpoint."""
  # Minimize RMSE and maximize AUROC
  compare_metric = operator.lt if regression else operator.gt
  # Calculate the AUROC/RMSE on the validation split
  validation_metric = metric_scores['test'](sess)
  tf1.logging.info('Epoch %d %s Val %.4f', epoch, metric_name,
                    validation_metric)
  if compare_metric(validation_metric, best_validation_metric):
    curr_best_epoch = epoch
    best_validation_metric = validation_metric
    best_train_metric = metric_scores['train'](sess)
    # copy the checkpoints files *.meta *.index, *.data* each time
    # there is a better result
    _update_latest_checkpoint(model_dir, best_checkpoint_dir)
  return curr_best_epoch, best_validation_metric, best_train_metric


def training(x_train, y_train, x_validation,
             y_validation, regression,
             logdir):
  """Trains the Neural Additive Model (NAM).

  Args:
    x_train: Training inputs.
    y_train: Training labels.
    x_validation: Validation inputs.
    y_validation: Validation labels.
    logdir: dir to save the checkpoints.

  Returns:
    Best train and validation evaluation metric obtained during NAM training.
  """
  tf1.logging.info('Started training with logdir %s', logdir)
  bs = min(batch_size, x_train.shape[0])
  num_steps_per_epoch = x_train.shape[0] // bs
  # Keep track of the best validation RMSE/AUROC and train AUROC score which
  # corresponds to the best validation metric score.
  if regression:
    best_train_metric = np.inf * np.ones(n_models)
    best_validation_metric = np.inf * np.ones(n_models)
  else:
    best_train_metric = np.zeros(n_models)
    best_validation_metric = np.zeros(n_models)
  # Set to a large value to avoid early stopping initially during training
  curr_best_epoch = np.full(n_models, np.inf)
  # Boolean variables to indicate whether the training of a specific model has
  # been early stopped.
  early_stopping = [False] * n_models
  # Classification: AUROC, Regression : RMSE Score
  metric_name = 'MSE' if regression else 'AUROC'
  tf1.reset_default_graph()
  with tf1.Graph().as_default():
    #tf1.compat.v1.set_random_seed(tf_seed)
    # Setup your training.
    graph_tensors_and_ops, metric_scores = _create_computation_graph(
        x_train, y_train, x_validation, y_validation, bs, regression)

    train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
        graph_tensors_and_ops, early_stopping)
    global_step = tf1.train.get_or_create_global_step()
    increment_global_step = tf1.assign(global_step, global_step + 1)
    saver_hooks, model_dirs, best_checkpoint_dirs = _create_graph_saver(
        graph_tensors_and_ops, logdir, num_steps_per_epoch)
    if debug:
      summary_writer = tf1.summary.FileWriter(os.path.join(logdir, 'tb_log'))

    with tf1.train.MonitoredSession(hooks=saver_hooks) as sess:
      for n in range(n_models):
        sess.run([
            graph_tensors_and_ops[n]['iterator_initializer'],
            graph_tensors_and_ops[n]['running_vars_initializer']
        ])
      for epoch in range(1, training_epochs + 1):
        if not all(early_stopping):
          for _ in range(num_steps_per_epoch):
            sess.run(train_ops)  # Train the network
          # Decay the learning rate by a fixed ratio every epoch
          sess.run(lr_decay_ops)
        else:
          tf1.logging.info('All models early stopped at epoch %d', epoch)
          break

        for n in range(n_models):
          if early_stopping[n]:
            sess.run(increment_global_step)
            continue
          # Log summaries
          if debug:
            global_summary, global_step = sess.run([
                graph_tensors_and_ops[n]['summary_op'],
                graph_tensors_and_ops[n]['global_step']
            ])
            summary_writer.add_summary(global_summary, global_step)

          if epoch % save_checkpoint_every_n_epochs == 0:
            (curr_best_epoch[n], best_validation_metric[n],
             best_train_metric[n]) = _update_metrics_and_checkpoints(
                 sess, epoch, metric_scores[n], curr_best_epoch[n],
                 best_validation_metric[n], best_train_metric[n], model_dirs[n],
                 best_checkpoint_dirs[n], regression, metric_name)
            if curr_best_epoch[n] + early_stopping_epochs < epoch:
              tf1.logging.info('Early stopping at epoch {}'.format(epoch))
              early_stopping[n] = True  # Set early stopping for model `n`.
              train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
                  graph_tensors_and_ops, early_stopping)
          # Reset running variable counters
          sess.run(graph_tensors_and_ops[n]['running_vars_initializer'])
          
  tf1.logging.info('Finished training.')
  for n in range(n_models):
    tf1.logging.info(
        'Model %d: Best Epoch %d, Individual %s: Train %.4f, Validation %.4f',
        n, curr_best_epoch[n], metric_name, best_train_metric[n],
        best_validation_metric[n])
  return np.mean(best_train_metric), np.mean(best_validation_metric)


def single_split_training(X_train, y_train, X_test, y_test, logdir, regression):
  from datetime import datetime
  tstart = datetime.now()
  err, err_test = training(X_train, y_train, X_test, y_test, regression, logdir)
  tend = datetime.now()

  training_seconds = (tend - tstart).seconds

  variables = {}

  if regression:
    variables["rmse"] = err
    variables["rmse_test"] = err_test
    variables["mse"] = err**2
    variables["mse_test"] = err_test**2
  else:
    variables["roc-auc"] = err
    variables["roc-auc_test"] = err_test
  
  variables["training_seconds"] = training_seconds
  
  return variables
  
if __name__ == '__main__':
    
  args = parser.parse_args()
  variables = vars(args)

  type = variables.get("type", None)
  distribution = variables["distribution"]
  family = variables["family"]    # gaussian / binomial
  iteration = variables["iteration"]

  conv_threshold = variables.pop("convergence_threshold", 0.01)
  delta_threshold = variables.pop("delta_threshold", 0.00001)

  variables.pop("iteration")
  print(variables)
  output_folder = variables.pop("output", "results")

  if iteration is not None:
    rel_path = "./{0}/{1}".format(output_folder,iteration)
    path = os.path.normpath(os.path.abspath(rel_path))
    #add iteration
    if not os.path.exists(path):
        os.makedirs(path)

  else:
    rel_path = "./{0}/".format(output_folder)
    path = os.path.normpath(os.path.abspath(rel_path))
      
  # add exec type
  data_type_path = "_".join(list(variables.values())) 
  path = os.path.normpath(os.path.join(path, data_type_path))
  if not os.path.exists(path):
    os.mkdir(path)

  logdir = path + "/model"
  print("Saving results on " + path)

  import pandas as pd
  import os.path as osp

  try:
    os.rmdir(logdir + "/model_0")
  except:
    pass

  X_train = pd.read_csv("./dataset/{0}/X_train.csv".format(data_type_path), index_col=0, dtype=np.float32).reset_index(drop=True)
  X_train.columns = X_train.columns.map(int)
  X_train = X_train.to_numpy()
  
  fs_train = pd.read_csv("./dataset/{0}/fs_train.csv".format(data_type_path), index_col=0).reset_index(drop=True)
  
  y_train = pd.read_csv("./dataset/{0}/y_train.csv".format(data_type_path), index_col=0, dtype=np.float32).reset_index(drop=True).squeeze()
  y_train = y_train.to_numpy()

  X_test = pd.read_csv("./dataset/{0}/X_test.csv".format(data_type_path), index_col=0, dtype=np.float32).reset_index(drop=True)
  X_test = X_test.to_numpy()

  fs_test = pd.read_csv("./dataset/{0}/fs_test.csv".format(data_type_path), index_col=0).reset_index(drop=True)
  
  y_test = pd.read_csv("./dataset/{0}/y_test.csv".format(data_type_path), index_col=0, dtype=np.float32).reset_index(drop=True).squeeze()
  y_test = y_test.to_numpy()

  regression = False
  if family == "gaussian":
    regression = True
    variables = single_split_training(X_train, y_train, X_test, y_test, logdir, regression)
  else:
    y_train_binomial = np.random.binomial(n=1, p=y_train, size=y_train.shape[0])
    y_test_binomial = np.random.binomial(n=1, p=y_test, size=y_test.shape[0])
    variables = single_split_training(X_train, y_train_binomial, X_test, y_test_binomial, logdir, regression)

    pd.DataFrame(y_train_binomial).to_csv(path + "/y_train_binomial.csv")
    pd.DataFrame(y_test_binomial).to_csv(path + "/y_test_binomial.csv")

  pd.DataFrame.from_dict(variables, orient="index").transpose().to_csv(path + "/variables_training.csv", index=False)  
  
  
    
