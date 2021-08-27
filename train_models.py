# from tensorflow.python.ops.gen_math_ops import xlogy_eager_fallback
from utils import save_weights_and_results
from models import *
import tensorflow as tf
import numpy as np
import hyperparams
import itertools  

_SMOKE_SIZE_ = 3000
# _SMOKE_SIZE_ = -1

## Define common training objects
epochs = 5
# epochs = 50

batch_size = 64
optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
early_stopping = tf.keras.callbacks.EarlyStopping(verbose=1, patience=3,monitor="val_out_layer_loss")
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1,monitor="val_out_layer_loss")

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


## DEFINE ALL MODELS
### FOR CYCLE ON TASKS
for current_task in hyperparams._TASKS_:
# for current_task in ['35kws']:

     ##  List that will contain all the initialized models
    all_models = []
    
    ## get filenames
    core_kws, aux_kws, output_classes = get_kws(hyperparams._DATA_DIR_, current_task)

    # if the binaries for the splits are not yet generated, generate them; otherwise just load them.
    if len(os.listdir(hyperparams._BINARIES_DIR_/current_task)) == 0:
        #Get train, validation and test data from the splits provided in the data directory
        make_and_save_original_splits(current_task, return_canonical_test_set=False)
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_original_splits(current_task, smoke_size=_SMOKE_SIZE_)
    
    ## GET TF DATASET
    train_dataset, train_steps, valid_dataset, valid_steps, test_dataset = get_tf_datasets(X_train,
        y_train, X_valid, y_valid, X_test, y_test, batch_size=batch_size, task=current_task, verbose=False)
    

    feature_types = ['mfccs']
    for ft_type in feature_types:
        
        # # define models
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

        for n_res_layers in [3,4,5]:
            resnet_andr = resnet_andreade(
                train_dataset,
                output_classes,
                model_suffix=f"{ft_type}_{n_res_layers}_layers", 
                n_res_blocks=n_res_layers,
                mfccs=True if ft_type =='mfccs' else False)
            
            all_models.append(resnet_andr)

        seq_q_no_cnn = andreade_seq_query_no_cnn(
                        train_dataset,
            output_classes,
            model_suffix = f"{ft_type}",
            mfccs=True if ft_type=='mfccs' else False)
        
        all_models.append(seq_q_no_cnn)

        #grid search on number of heads for MHA
        for n_heads in [2,3,4,5]:
            andreade_mha = mha_andreade(
                train_dataset,
                output_classes,
                model_suffix=f"{ft_type}_{n_heads}heads",
                mfccs=True if ft_type=='mfccs' else False,
                n_heads=n_heads)

            all_models.append(andreade_mha)

        ## KWT Test different  n_heads-------------------

        # for n_layers in [2,4,6,8]:
        #     kwt = KWT(train_dataset, 
        #             model_suffix = f"{ft_type}_{n_layers}layers",
        #             num_patches=98,
        #             num_layers=n_layers,
        #             d_model=192,
        #             num_heads=3,
        #             mlp_dim=768,
        #             output_classes=output_classes)
        #     all_models.append(kwt)
        

    print([model.name for model in all_models])

    for model in all_models:
        print(f"---------- CURRENT TASK: {current_task} ----------")
        # print(f"----------- TRAINING {model.name} ----------------")
        print(model.summary())
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
        
        save_weights_and_results(trained_model, history, current_task)

        

    
