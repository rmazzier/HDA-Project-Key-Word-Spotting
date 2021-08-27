import os
import librosa.display
import librosa as lr
from hyperparams import _MODELS_DIR_
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from custom_layers import *

import matplotlib.pyplot as plt
import numpy as np


def show_spectrogram(spectrogram):

    plt.figure(figsize=(17, 10))
    plt.subplot(2, 1, 1)
    lr.display.specshow(spectrogram, sr=16000,
                        y_axis='linear', x_axis="time", hop_length=160)
    plt.colorbar()
    plt.title('Default magnitude Spectrum (as outputted by the Spectrogram Layer)')

    plt.subplot(2, 1, 2)
    lr.display.specshow(lr.amplitude_to_db(spectrogram, ref=np.max),
                        sr=16000,
                        y_axis='linear',
                        x_axis='time',
                        hop_length=160)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram converted to db scale")
    plt.tight_layout()


def train_and_save_model(model, train_data, valid_data, train_steps, valid_steps, epochs=50):
    # Check if the model is saved, in that case load it. Otherwise train it.
    if model.name+'.h5' in os.listdir(_MODELS_DIR_):
        # load model
        model = load_model(_MODELS_DIR_/(model.name+'.h5'),
                           custom_objects={"MFCC": MFCC})
        # load training data
        with open(f'models/history_{model.name}.pkl', 'rb') as inp:
            training_data = pickle.load(inp)
        print(f"""Model was already saved, therefore I loaded the model. If you want to retrain the model from scratch, delete the model files from {_MODELS_DIR_} directory""")
    else:
        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(
            verbose=1, patience=3)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
        history = model.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr],
            steps_per_epoch=train_steps,
            validation_steps=valid_steps)

        # save model
        model.save(_MODELS_DIR_/(model.name+'.h5'))
        # save training data
        with open(_MODELS_DIR_/f'history_{model.name}.pkl', 'wb') as outp:
            pickle.dump(history.history, outp, pickle.HIGHEST_PROTOCOL)
        training_data = history.history
    return model, training_data

def visualize_class_balance(y_train, y_test, output_classes):
    #visualize class balance
    unique_tr, counts_tr = np.unique(y_train, return_counts=True)
    unique_te, counts_te = np.unique(y_test, return_counts=True)

    #display same class balance
    fig, axes = plt.subplots(2, figsize=(22,10))
    axes[0].bar([output_classes[i] for i in  unique_tr], counts_tr, alpha=0.6)
    axes[1].bar([output_classes[i] for i in  unique_te], counts_te, alpha=0.6)
    axes[0].set_ylabel("Amount of samples")
    axes[1].set_ylabel("Amount of samples")
    axes[0].set_title("Training set class percentage")
    axes[1].set_title("Test set class percentage")
    plt.xticks(rotation=45)
    plt.show()

def save_weights_and_results(model, history, current_task):
    #save model weights
    model.save_weights(hyperparams._MODELS_DIR_/current_task/model.name/"weights")
    results={}
    #save attention scores and predictions
    # y_scores, att_scores = model.predict(test_dataset.batch(len(X_test)))
    # y_pred = np.array(np.argmax(y_scores, axis=1))
    # y_true = np.array(y_test)
    # compute test accuracy
    # test_acc = sum(np.equal(y_pred, y_true)) / len(y_true)
    # print(f'Test set accuracy: {test_acc:.3%}')
    
    # results["test_acc"] = test_acc
    # results["attention_scores"] = att_scores
    # results["prediction_scores"] = y_scores
    results["train_history"] = history.history
    with open(hyperparams._MODELS_DIR_/current_task/model.name/f'train_history.pkl', 'wb') as outp:
        pickle.dump(results, outp)
    
def get_n_of_trainable_variables(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])