## Runnning Neural Additive Model (NAM)

The main folder contains the base scripts. They can be modified to set up different simulation scenarios and/or NAM architectures. 

The sub-folders Scenario I and Scenario II contain the .py and run.sh files to launch each simulation scenario. 

To run in CESGA a single simulation, execute the command: 

```
sbatch -a 1-1 -t 03:00:00 run.sh &
```

To run 1000 simulations, you can user run_cont.sh script: 

```
sh run_cont.sh > run-out.out 2>&1 &
```

## Scenario I 


```
training_epochs = 1000
learning_rate = 0.00674
output_regularization =  0.0
l2_regularization = 1e-6
batch_size = 1024
decay_rate = 0.995
dropout = 0.0
feature_dropout = 0.0
num_basis_functions = 1024
shallow = True
units_multiplier = 1
cross_val = False
max_checkpoints_to_keep = 1
save_checkpoint_every_n_epochs = 10
n_models = 1
fold_num = 1
activation = "exu"
debug = False
use_dnn = False
early_stopping_epochs = 100
```

## Scenario II - Modified graph_builder

In order to specify Deep NN with a particular size, I modified the `graph_builder.py` code to accept a list of sizes. If the requirements are re-installed, you might need to copy again the file to `env/src/neural-additive-models/neural_additive_models/graph_builder.py`. 

```
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
```


