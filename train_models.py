from tensorflow.python.ops.gen_math_ops import xlogy_eager_fallback
from models import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import hyperparams

## Define common training objects
epochs = 50
batch_size = 64
optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
early_stopping = tf.keras.callbacks.EarlyStopping(verbose=1, patience=4)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1)

_SMOKE_SIZE_ = 2000

### FOR CYCLE ON TASKS
for current_task in hyperparams._TASKS_:
    print(f"---------- TRAINING FOR TASK: {current_task} ----------")

    ## get filenames
    core_kws, aux_kws, output_classes = get_kws(hyperparams._DATA_DIR_, current_task)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_original_splits(current_task, smoke_size=_SMOKE_SIZE_)
    
    
    ## GET TF DATASET
    train_dataset, train_steps, valid_dataset, valid_steps, test_dataset = get_tf_datasets(X_train,
        y_train, X_valid, y_valid, X_test, 
        y_test, batch_size=batch_size, task=current_task)

    ## for cycle on number of transformer layers
    for n_transformer_layers in range(1,13):
        pass #TODO
    
