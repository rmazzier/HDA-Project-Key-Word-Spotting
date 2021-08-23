# from tensorflow.python.ops.gen_math_ops import xlogy_eager_fallback
from models import *
import tensorflow as tf
import numpy as np
import hyperparams
import itertools  

## Define common training objects
epochs = 30
batch_size = 64
optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
early_stopping = tf.keras.callbacks.EarlyStopping(verbose=1, patience=4)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1)

def compile_and_train_model(model, train_dataset, valid_dataset, epochs, optimizer, loss, early_stopping, reduce_lr, train_steps, valid_steps):
    
    model.compile(
        optimizer=optimizer,
        loss={'out_layer':loss},
        metrics={'out_layer':'accuracy'},
    )

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        steps_per_epoch=train_steps,
        validation_steps=valid_steps)
    
    return model, history

_SMOKE_SIZE_ = -1

## DEFINE ALL MODELS
### FOR CYCLE ON TASKS
for current_task in hyperparams._TASKS_:

     ##  List that will contain all the initialized models
    all_models = []
    
    ## get filenames
    core_kws, aux_kws, output_classes = get_kws(hyperparams._DATA_DIR_, current_task)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_original_splits(current_task, smoke_size=_SMOKE_SIZE_)
    
    
    ## GET TF DATASET
    train_dataset, train_steps, valid_dataset, valid_steps, test_dataset = get_tf_datasets(X_train,
        y_train, X_valid, y_valid, X_test, 
        y_test, batch_size=batch_size, task=current_task, verbose=True)

    ## for cycle on the features types
    # mfccs = 40 Cepstral Coefficients
    # log_mel = 80 Log Mel Features

    feature_types = ['mfccs', 'log_mel']
    for ft_type in feature_types:
        
        # define models
        att_rnn = simple_attention_rnn(
            train_dataset,
            output_classes,
            model_suffix=f"{ft_type}",
            mfccs=True if ft_type == 'mfccs' else False)

        all_models.append(att_rnn)

        andreade_original = attention_rnn_andreade(
            train_dataset, 
            output_classes, 
            model_suffix=f"{ft_type}",
            mfccs= True if ft_type=='mfccs' else False)
        
        all_models.append(andreade_original)
        
        andreade_queries = attention_rnn_andreade_seq_query(
                train_dataset,
                output_classes,
                model_suffix = f"{ft_type}",
                mfccs=True if ft_type=='mfccs' else False)

        all_models.append(andreade_queries)
        
        # grid search on filter dimensions
        f_widths=[5]
        f_heights=[1]
        for fw,fh in list(itertools.product(f_widths, f_heights)):
            print("Current filter dimension:",(fw,fh))
            #grid search on number of heads for MHA
            for n_heads in [7]:
                andreade_mha = mha_andreade(
                    train_dataset,
                    output_classes,
                    model_suffix=f"{ft_type}_{fw}{fh}filter_{n_heads}heads",
                    mfccs=True if ft_type=='mfccs' else False,
                    n_heads=n_heads,
                    filter_w = fw,
                    filter_h = fh)

                all_models.append(andreade_mha)

    print([model.name for model in all_models])

    for model in all_models:
        print(f"---------- CURRENT TASK: {current_task} ----------")
        print(f"----------- TRAINING {model.name} ----------------")
        trained_model, history = compile_and_train_model(
            model, 
            train_dataset, 
            valid_dataset,
            epochs=epochs,
            optimizer=optimizer,
            loss=loss,
            early_stopping=early_stopping,
            reduce_lr = reduce_lr,
            train_steps=train_steps,
            valid_steps=valid_steps)

        y_scores = trained_model.predict(test_dataset)[0]
        y_pred = np.array(np.argmax(y_scores, axis=1))
        y_true = np.array(y_test)

        # compute test accuracy
        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.3%}')
        trained_model.save_weights(hyperparams._MODELS_DIR_/current_task/trained_model.name/f"{trained_model.name}_weights")

    
