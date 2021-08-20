
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

from input_pipeline import _DATA_DIR_, _BINARIES_DIR_, _UNKNOWN_CLASS_, _SILENCE_CLASS_, _MODELS_DIR_, _TASKS_
from input_pipeline import *
from metrics import *
from custom_layers import MFCC, LogMelSpectrogram


def cnn_trad_fpool3(ds, output_classes, n_mfcc, mfcc_deltas, model_name, fft_size=512, win_size=400, hop_size=160, n_filters=40):
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = MFCC(sample_rate=16000,
             fft_size=fft_size,
             win_size=win_size,
             hop_size=hop_size,
             n_filters=n_filters,
             n_cepstral=n_mfcc,
             return_deltas=mfcc_deltas)(X_input)
    X = layers.BatchNormalization(axis=-1)(X)
    X = layers.Conv2D(64, (20, 8), activation='relu')(X)
    X = layers.MaxPool2D(pool_size=(1, 3))(X)
    X = layers.Conv2D(64, (10, 4), activation='relu')(X)
    X = layers.Flatten()(X)
    X = layers.Dense(32)(X)
    X = layers.Dense(128, activation='relu')(X)
    X = layers.Dense(len(output_classes))(X)

    model = tf.keras.Model(inputs=X_input, outputs=X, name=model_name)
    return model

def cnn_one_fstride4(ds, output_classes, n_mfcc, mfcc_deltas, model_name, fft_size=512, win_size=400, hop_size=160, n_filters=40):
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = MFCC(sample_rate = 16000, 
             fft_size=fft_size, 
             win_size=win_size, 
             hop_size=hop_size, 
             n_filters=n_filters, 
             n_cepstral=n_mfcc,
            return_deltas=mfcc_deltas)(X_input)
    X = layers.BatchNormalization(axis=-1)(X)

    X = layers.Conv2D(186, (32, 8), strides=(1, 4), activation='relu')(X)
    X = layers.Flatten()(X)
    X = layers.Dense(32)(X)
    X = layers.Dense(128, activation='relu')(X)
    X = layers.Dense(128, activation='relu')(X)
    X = layers.Dense(len(output_classes))(X)

    model = tf.keras.Model(inputs=X_input, outputs=X, name=model_name)
    return model

def kws_res_net(ds):
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)
    
    X_input0 = tf.keras.Input(input_shape)
    #X = MelSpectrogram(sample_rate = 16000, fft_size=512, win_size=400, hop_size=160, n_filters=40)(X_input)
    X_input = MFCC(sample_rate = 16000, fft_size=512, win_size=400, hop_size=160, n_filters=40, n_cepstral=40)(X_input0)
    X = layers.BatchNormalization(axis=-1)(X_input)
    
    # Stage 1 (4 lines)
    X = layers.Conv2D(64, (20,8), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis =-1, name = 'bn_conv1')(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((1, 3))(X)

    # Stage 2 (3 lines)
    X = convolutional_block(X, f = 3, filters = [64, 64, 64], stage = 2, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 64], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 64], stage=2, block='c')

    # AVGPOOL (1 line). Use "X = AveragePooling2D(...)(X)"
    X = tf.keras.layers.AveragePooling2D(name='avg_pool')(X)

    # Output layer
    X = layers.Flatten()(X)
    X = layers.Dense(len(output_classes), name='fc', kernel_initializer = glorot_uniform(seed=0))(X)
    
    ### END CODE HERE ###
    
    # Create model
    model = tf.keras.Model(inputs = X_input0, outputs = X, name='ResNetKWS')

    return model

def simple_rnn(ds, 
               output_classes, 
               n_mfcc, 
               mfcc_deltas, 
               fft_size=512, 
               win_size=400, 
               hop_size=160, 
               n_filters=40):
    
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = MFCC(sample_rate = 16000, 
             fft_size=fft_size, 
             win_size=win_size, 
             hop_size=hop_size, 
             n_filters=n_filters, 
             n_cepstral=n_mfcc,
            return_deltas=mfcc_deltas)(X_input)
    X = layers.BatchNormalization(axis=-1)(X)
    
    X = layers.Lambda(lambda w: tf.keras.backend.squeeze(w, -1))(X)
    X = layers.Bidirectional(layers.GRU(units=128))(X)
    X = layers.Dense(len(output_classes))(X)

    model = tf.keras.Model(inputs=X_input, outputs=X, name="SimpleRNN")
    return model

def cnn_rnn(ds, 
               output_classes, 
               n_mfcc, 
               mfcc_deltas, 
               fft_size=512, 
               win_size=400, 
               hop_size=160, 
               n_filters=40):
    
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = MFCC(sample_rate = 16000, 
             fft_size=fft_size, 
             win_size=win_size, 
             hop_size=hop_size, 
             n_filters=n_filters, 
             n_cepstral=n_mfcc,
            return_deltas=mfcc_deltas)(X_input)
    X = layers.BatchNormalization(axis=-1)(X)
    
    X = layers.Conv2D(64, (20,1))(X)
    X = layers.BatchNormalization(axis =-1)(X)
    X = layers.Activation('relu')(X)
    
    X = layers.Conv2D(1, (10,1))(X)
    X = layers.BatchNormalization(axis =-1)(X)
    X = layers.Activation('relu')(X)

    
    X = layers.Lambda(lambda w: tf.keras.backend.squeeze(w, -1))(X)
    X = layers.Bidirectional(layers.GRU(units=128))(X)
    X = layers.Dense(len(output_classes))(X)

    model = tf.keras.Model(inputs=X_input, outputs=X, name="SimpleRNN")
    return model

def simple_attention_rnn(ds):
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = MFCC(sample_rate = 16000, 
                 fft_size=512, 
                 win_size=400, 
                 hop_size=160, 
                 n_filters=40, 
                 n_cepstral=13,
                return_deltas=True)(X_input)
    X = layers.Lambda(lambda x : x[...,-1], name="squeeze_channel_dimension")(X)
    X = layers.Bidirectional(layers.GRU(units=64, return_sequences=True), name="BidirectionalGRU")(X)
    last_out = layers.Lambda(lambda x: x[:,-1,:])(X)
    Q = layers.Dense(128)(last_out)
    Q = layers.Lambda(lambda x: tf.expand_dims(x, 1))(Q)
    weighted_seq, att_ws = layers.Attention()([Q, X], return_attention_scores=True)

    O = layers.Dense(128, activation='relu')(weighted_seq)
    O = layers.Dense(64, activation='relu')(O)
    O = layers.Dense(len(output_classes), name="out_layer")(O)

    att_model = tf.keras.Model(inputs = [X_input], outputs=[O,att_ws])
    return att_model

def attention_rnn_andreade(ds):
    """Neural attention model proposed in de Andreade et al. 2018"""
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = LogMelSpectrogram(sample_rate = 16000, 
                 fft_size=1024, 
                 win_size=400, 
                 hop_size=160, 
                 n_filters=80)(X_input)
    
    # CNN part
    X = layers.Conv2D(10, (5, 1), activation='relu', padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Conv2D(1, (5, 1), activation='relu', padding='same')(X)
    X = layers.BatchNormalization()(X)
    
    # Recurrent Part
    X = layers.Lambda(lambda x : x[...,-1], name="squeeze_channel_dimension")(X)
    X = layers.Bidirectional(layers.GRU(units=64, return_sequences=True), name="BidirectionalGRU")(X)
    last_out = layers.Lambda(lambda x: x[:,-1,:])(X)
    
    # Self-Attention
    Q = layers.Dense(128)(last_out)
    Q = layers.Lambda(lambda x: tf.expand_dims(x, 1))(Q)
    weighted_seq, att_ws = layers.Attention()([Q, X], return_attention_scores=True)

    O = layers.Dense(64, activation='relu')(weighted_seq)
    O = layers.Dense(len(output_classes), name="out_layer")(O)

    att_model = tf.keras.Model(inputs = [X_input], outputs=[O,att_ws])
    return att_model

# fare alternativa con multi head attention!!

if __name__ == "__main__":
    ## Train all the CNN Networks on all the different tasks.
    n_mfcc=40
    n_filters = 40
    mfcc_deltas=False
    win_size = 400
    hop_size=160

    # Set to -1 to train with all data
    _SMOKE_SIZE_ = 1000


    for current_task in _TASKS_:
        print(f"---------- CURRENT TASK: {current_task}")
        core_kws, aux_kws, output_classes = get_kws(_DATA_DIR_, current_task)
        # print("Core keywords: ", core_kws)
        # print()
        # print("Auxiliary keywords: ", aux_kws)
        # print()
        print("Output Classes: ", output_classes)
        # if the binaries for the splits are not yet generated, generate them; otherwise just load them.
        if len(os.listdir(_BINARIES_DIR_/current_task)) == 0:
            #Get train, validation and test data from the splits provided in the data directory
            X_train, y_train, X_valid, y_valid, X_test, y_test = get_original_splits(current_task)
        else:
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_original_splits(current_task)

        X_train, y_train, X_valid, y_valid, X_test, y_test = get_smoke_sized(X_train, 
                                                                            y_train, 
                                                                            X_valid, 
                                                                            y_valid, 
                                                                            X_test, 
                                                                            y_test, 
                                                                            smoke_size=_SMOKE_SIZE_)

        # print(f"Samples in Training Set: {len(X_train)}")
        # print(f"Samples in Test Set: {len(X_test)}")
        # print(f"Samples in Validation Set: {len(X_valid)}")

        batch_size = 64
        train_dataset, train_steps, valid_dataset, valid_steps, test_dataset, test_steps = get_tf_datasets(X_train, 
                                    y_train, 
                                    X_valid, 
                                    y_valid, 
                                    X_test, 
                                    y_test, 
                                    batch_size=batch_size)

        ## models definition
        ######## cnn_trad_fpool3 ################
        model_name = "cnn_trad_fpool3"
        model_cnn_trad_fpool3 = cnn_trad_fpool3(train_dataset,
                                                output_classes,
                                                n_mfcc=n_mfcc,
                                                mfcc_deltas=mfcc_deltas,
                                            model_name = model_name,
                                            n_filters=n_filters,
                                                win_size=win_size,
                                                hop_size=hop_size)
        model_cnn_trad_fpool3.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, patience=4)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', verbose=1, patience=3)
        history = model_cnn_trad_fpool3.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=30,
            callbacks=[early_stopping, reduce_lr],
            steps_per_epoch=train_steps,
            validation_steps=valid_steps)

        #save model
        #cnn_trad_fpool3
        model_cnn_trad_fpool3.save(_MODELS_DIR_/current_task/(model_cnn_trad_fpool3.name+'.h5'))

        ## trad_one_fstride4
        model_name = "cnn_one_fstride4"

        model_cnn_one_fstride4 = cnn_one_fstride4(train_dataset,
                                                output_classes,
                                                n_mfcc=n_mfcc,
                                                mfcc_deltas=mfcc_deltas,
                                            model_name = model_name,
                                            n_filters=n_filters,
                                                win_size=win_size,
                                                hop_size=hop_size)
        model_cnn_one_fstride4.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, patience=3)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
        model_cnn_one_fstride4.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=30,
            callbacks=[early_stopping, reduce_lr],
            steps_per_epoch=train_steps,
            validation_steps=valid_steps)

        #cnn_one_fst
        model_cnn_one_fstride4.save(_MODELS_DIR_/current_task/(model_cnn_one_fstride4.name+'.h5'))

        # ResNet
        model_res_net = kws_res_net(train_dataset)

        model_res_net.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, patience=3)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
        history = model_res_net.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=30,
            callbacks=[early_stopping, reduce_lr],
            steps_per_epoch=train_steps,
            validation_steps=valid_steps)

        # ResNet1
        model_res_net.save(_MODELS_DIR_/current_task/(model_res_net.name+'.h5'))