
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

from hyperparams import _DATA_DIR_, _BINARIES_DIR_, _UNKNOWN_CLASS_, _SILENCE_CLASS_, _MODELS_DIR_, _TASKS_
from input_pipeline import *
from metrics import *
from custom_layers import *


def cnn_trad_fpool3(ds, output_classes, model_suffix):
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = MFCC()(X_input)
    X = layers.BatchNormalization(axis=-1)(X)
    X = SpecAugment()(X)
    X = layers.Conv2D(64, (20, 8), activation='relu')(X)
    X = layers.MaxPool2D(pool_size=(1, 3))(X)
    X = layers.Conv2D(64, (10, 4), activation='relu')(X)
    X = layers.Flatten()(X)
    X = layers.Dense(32)(X)
    X = layers.Dense(128, activation='relu')(X)
    X = layers.Dense(len(output_classes))(X)

    model = tf.keras.Model(inputs=X_input, outputs=X, name='cnn_trad_fpool3_'+model_suffix)
    return model

def cnn_one_fstride4(ds, output_classes, model_suffix):
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = MFCC()(X_input)
    X = layers.BatchNormalization(axis=-1)(X)

    X = layers.Conv2D(186, (32, 8), strides=(1, 4), activation='relu')(X)
    X = layers.Flatten()(X)
    X = layers.Dense(32)(X)
    X = layers.Dense(128, activation='relu')(X)
    X = layers.Dense(128, activation='relu')(X)
    X = layers.Dense(len(output_classes))(X)

    model = tf.keras.Model(inputs=X_input, outputs=X, name="cnn_one_fstride4_"+model_suffix)
    return model

def kws_res_net(ds, output_classes, model_suffix):
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)
    
    X_input = tf.keras.Input(input_shape)
    #X = MelSpectrogram(sample_rate = 16000, fft_size=512, win_size=400, hop_size=160, n_filters=40)(X_input)
    X = MFCC()(X_input)
    X = SpecAugment()(X)
    X = layers.BatchNormalization(axis=-1)(X)
    
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
    model = tf.keras.Model(inputs = X_input, outputs = X, name="kws_res_net"+model_suffix)

    return model

def simple_rnn(ds, output_classes, model_suffix, mfccs=True):
    
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = RandomNoiseAugment()(X_input)
    if mfccs:
        X = MFCC()(X)
    else:
        X = LogMelSpectrogram()(X)
        X = layers.Lambda(lambda x : x[:,:,1:,:], name="remove_energies")(X)
        
    X = SpecAugment()(X)
    X = layers.BatchNormalization(axis=-1)(X)
    
    X = layers.Lambda(lambda w: tf.keras.backend.squeeze(w, -1))(X)
    X = layers.Bidirectional(layers.GRU(units=128))(X)
    X = layers.Dense(len(output_classes))(X)

    model = tf.keras.Model(inputs=X_input, outputs=X, name="simple_rnn_"+model_suffix)
    return model

# def cnn_rnn(ds, output_classes, model_suffix):
    
#     for s, _ in ds.take(1):
#         input_shape = s.shape[1:]
#         print('Input shape:', input_shape)

#     X_input = tf.keras.Input(input_shape)
#     X = MFCC()(X_input)
#     X = layers.BatchNormalization(axis=-1)(X)
    
#     X = layers.Conv2D(64, (20,1))(X)
#     X = layers.BatchNormalization(axis =-1)(X)
#     X = layers.Activation('relu')(X)
    
#     X = layers.Conv2D(1, (10,1))(X)
#     X = layers.BatchNormalization(axis =-1)(X)
#     X = layers.Activation('relu')(X)

    
#     X = layers.Lambda(lambda w: tf.keras.backend.squeeze(w, -1))(X)
#     X = layers.Bidirectional(layers.GRU(units=128))(X)
#     X = layers.Dense(len(output_classes))(X)

#     model = tf.keras.Model(inputs=X_input, outputs=X, name="cnn_rnn_"+model_suffix)
#     return model

def simple_attention_rnn(ds, output_classes, model_suffix, mfccs=True):
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = RandomNoiseAugment()(X_input)
    if mfccs:
        X = MFCC()(X)
    else:
        X = LogMelSpectrogram()(X)
        X = layers.Lambda(lambda x : x[:,:,1:,:], name="remove_energies")(X)
    
    X = SpecAugment()(X)

    X = layers.Lambda(lambda x : x[...,-1], name="squeeze_channel_dimension")(X)
    X = layers.Bidirectional(layers.GRU(units=64, return_sequences=True), name="BidirectionalGRU")(X)
    last_out = layers.Lambda(lambda x: x[:,-1,:])(X)
    Q = layers.Dense(128)(last_out)
    Q = layers.Lambda(lambda x: tf.expand_dims(x, 1))(Q)
    weighted_seq, att_ws = layers.Attention()([Q, X], return_attention_scores=True)
    weighted_seq = layers.Lambda(lambda x: x[:,0,:])(weighted_seq)

    O = layers.Dense(128, activation='relu')(weighted_seq)
    O = layers.Dense(64, activation='relu')(O)
    O = layers.Dense(len(output_classes), name="out_layer")(O)

    att_model = tf.keras.Model(inputs = [X_input], outputs=[O,att_ws], name="simple_attention_rnn_"+model_suffix)
    return att_model

def attention_rnn_andreade(ds, output_classes, model_suffix, mfccs=True):
    """Neural attention model proposed in de Andreade et al. 2018"""
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = RandomNoiseAugment()(X_input)
    if mfccs:
        X = MFCC()(X)
    else:
        X = LogMelSpectrogram()(X)
        X = layers.Lambda(lambda x : x[:,:,1:,:], name="remove_energies")(X)
    
    X = SpecAugment()(X)
    X = layers.BatchNormalization()(X)
    # CNN part
    X = layers.Conv2D(10, (5, 1), activation='relu', padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Conv2D(1, (5, 1), activation='relu', padding='same')(X)
    X = layers.BatchNormalization()(X)
    
    # Recurrent Part
    X = layers.Lambda(lambda x : x[...,-1], name="squeeze_channel_dimension")(X)
    X = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True), name="BidirectionalLSTM")(X)
    X = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True), name="BidirectionalLSTM2")(X)
    last_out = layers.Lambda(lambda x: x[:,-1,:])(X)
    
    # Self-Attention
    Q = layers.Dense(128)(last_out)
    Q = layers.Lambda(lambda x: tf.expand_dims(x, 1))(Q)
    weighted_seq, att_ws = layers.Attention()([Q, X], return_attention_scores=True)

    weighted_seq = layers.Lambda(lambda x: x[:,0,:])(weighted_seq)
    O = layers.Dense(64, activation='relu')(weighted_seq)
    O = layers.Dense(len(output_classes), name="out_layer")(O)

    att_model = tf.keras.Model(inputs = [X_input], outputs=[O,att_ws], name="andreade_original_"+model_suffix)
    return att_model

def attention_rnn_andreade_seq_query(ds, output_classes, model_suffix, mfccs=True, filter_w=5, filter_h=1):
    """Neural attention model proposed in de Andreade et al. 2018 with more queries. Also using GRU units
    Motivation: we know that in encoder/decoder models for machine translation, relying only on the last state
    of the encoder """
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = RandomNoiseAugment()(X_input)
    if mfccs:
        X = MFCC()(X)
    else:
        X = LogMelSpectrogram()(X)
        X = layers.Lambda(lambda x : x[:,:,1:,:], name="remove_energies")(X)
    
    X = SpecAugment()(X)
    # CNN part
    X = layers.Conv2D(10, (filter_w, filter_h), activation='relu', padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Conv2D(1, (filter_w, filter_h), activation='relu', padding='same')(X)
    X = layers.BatchNormalization()(X)
    
    # Recurrent Part
    X = layers.Lambda(lambda x : x[...,-1], name="squeeze_channel_dimension")(X)
    X = layers.Bidirectional(layers.GRU(units=64, return_sequences=True), name="BidirectionalGRU")(X)
    X = layers.Bidirectional(layers.GRU(units=64, return_sequences=True), name="BidirectionalGRU2")(X)

    # Self-Attention
#     Q = layers.Dense(128)(last_out)
    Q = layers.Dense(128)(X)
#     Q = layers.Lambda(lambda x: tf.expand_dims(x, 1))(Q)
    weighted_seq, att_ws = layers.Attention()([Q, X], return_attention_scores=True)
    
    X = layers.Bidirectional(layers.GRU(units=32), name="BidirectionalGRU3")(weighted_seq)

    O = layers.Dense(64, activation='relu')(X)
    O = layers.Dense(len(output_classes), name="out_layer")(O)

    att_model = tf.keras.Model(inputs = [X_input], outputs=[O,att_ws], name = "andreade_seq_query_"+model_suffix)
    return att_model


def mha_andreade(ds, output_classes, model_suffix, mfccs=True, n_heads=7, mha_encoding_dim=64, filter_w=5, filter_h=1):
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)

    X = RandomNoiseAugment()(X_input)
    if mfccs:
        X = MFCC()(X)
    else:
        X = LogMelSpectrogram()(X)
        X = layers.Lambda(lambda x : x[:,:,1:,:], name="remove_energies")(X)
    
    X = SpecAugment()(X)
    
    # CNN part
    X = layers.Conv2D(10, (filter_w, filter_h), activation='relu', padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Conv2D(1, (filter_w, filter_h), activation='relu', padding='same')(X)
    X = layers.BatchNormalization()(X)
    
    # Recurrent Part
    X = layers.Lambda(lambda x : x[...,-1], name="squeeze_channel_dimension")(X)
    X = layers.Bidirectional(layers.GRU(units=64, return_sequences=True), name="BidirectionalGRU")(X)
    X = layers.Bidirectional(layers.GRU(units=64, return_sequences=True), name="BidirectionalGRU2")(X)
    # last_out = layers.Lambda(lambda x: x[:,-1,:])(X)
    
    # Self-Multi-Headed Attention
    Q = layers.Dense(128)(X)
    # Q = layers.Lambda(lambda x: tf.expand_dims(x, 1))(Q)
    weighted_seq, att_ws = layers.MultiHeadAttention(num_heads=n_heads, key_dim=mha_encoding_dim)(Q, X, return_attention_scores=True)

    X = layers.Bidirectional(layers.GRU(units=32), name="BidirectionalGRU3")(weighted_seq)

    O = layers.Dense(64, activation='relu')(X)
    O = layers.Dense(len(output_classes), name="out_layer")(O)

    att_model = tf.keras.Model(inputs = [X_input], outputs=[O,att_ws], name = "mha_andreade_"+model_suffix)
    return att_model

def KWT(ds, 
        num_patches,
        num_layers,
        d_model,
        num_heads,
        mlp_dim,
        output_classes,
        model_suffix,
        dropout=0.1,
        mfccs=True):
    for s, _ in ds.take(1):
        input_shape = s.shape[1:]
        print('Input shape:', input_shape)

    X_input = tf.keras.Input(input_shape)
    X = RandomNoiseAugment()(X_input)
    if mfccs:
        X = MFCC()(X)
    else:
        X = LogMelSpectrogram()(X)
        X = layers.Lambda(lambda x : x[:,:,1:,:], name="remove_energies")(X)
    
    X = SpecAugment()(X)
    
    #remove channel dimension
    X = layers.Lambda(lambda x : x[...,0], name="removeChannelDimension")(X)
    
    # projection of patches
    # X = layer.Dense(...)(X)
    X = layers.Dense(d_model,
                     kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02), 
                     bias_initializer=tf.keras.initializers.Zeros())(X)
    
    # Apply Positional and Class embedding
    X = PosAndClassEmbed(num_patches, d_model)(X)
    
    #for cycle on TransformerBlocks
    transf_layers = [TransformerBlock(d_model, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
    for layer in transf_layers:
        X, _ = layer(X)
    
    # First (class token) is used for classification,
    class_output = layers.Lambda(lambda x : x[:,0], name="getClassToken")(X)
    O = layers.Dense(len(output_classes), name="out_layer")(class_output)

    model = tf.keras.Model(inputs = [X_input], outputs=[O], name="KWT_"+model_suffix)
    return model


if __name__ == "__main__":
    pass

