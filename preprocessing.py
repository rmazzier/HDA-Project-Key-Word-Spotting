import os
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank


def get_kws(data_dir):
    """Scans the data directory and outputs the core keywords (the ones that our NN will have to learn to classify),
    the auxiliary keywords (the ones to be classified in the "filler" class), and the list of actual output classes
    for our model."""
    commands = np.array(tf.io.gfile.listdir(data_dir))
    commands = commands[commands != 'README.md']
    commands = commands[commands != 'LICENSE']
    commands = commands[commands != 'testing_list.txt']
    commands = commands[commands != 'validation_list.txt']
    commands = commands[commands != '_background_noise_']
    commands = commands[commands != '.DS_Store']
    core_kws = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop",
                "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    aux_kws = list(set(commands) - set(core_kws))
    output_classes = core_kws+['filler']
    return core_kws, aux_kws, output_classes


def get_filenames(data_dir, commands):
    """Scans each directory inside the data directory for all the audio files. 
    Returns a list with all the audio files names"""
    filenames = []
    for command in commands:
        filenames += tf.io.gfile.glob(str(data_dir) +
                                      os.path.sep + command + '/*')
    return np.array(filenames)


def get_label_int(audio_file_path, aux_kws, output_classes):
    """Input: path of the audio file;
       Output: integer containing index of keyword associated to the audio file. """
    file_path_names = audio_file_path.split(os.path.sep)
    kw = file_path_names[2]
    if kw in aux_kws:
        return tf.argmax('filler' == np.array(output_classes))
    else:
        return tf.argmax(kw == np.array(output_classes))


def get_train_valid_test_split(filenames, labels, train_percentage, valid_percentage, test_percentage, seed=0):
    """Split the whole dataset into train, validation and test split, keeping for each the same proportion among output classes."""
    try:
        assert train_percentage + valid_percentage + test_percentage == 1
    except AssertionError:
        raise(AssertionError("Sum of train, valid and test percentage must be 1."))

    n = len(filenames)
    n_train = int(n * train_percentage)
    n_valid = int(n * valid_percentage)
    n_test = n - n_train - n_valid

    # (train+valid) - test split
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=n_test, random_state=seed)
    tv_indexes, test_indexes = next(sss.split(filenames, labels))

    tv_set = filenames[tv_indexes]
    tv_labels = labels[tv_indexes]
    X_test = filenames[test_indexes]
    y_test = labels[test_indexes]

    # Train - validation split
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=n_valid, random_state=seed)
    train_indexes, valid_indexes = next(sss2.split(tv_set, tv_labels))

    X_train = tv_set[train_indexes]
    y_train = tv_labels[train_indexes]
    X_valid = tv_set[valid_indexes]
    y_valid = tv_labels[valid_indexes]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def decode_audio(audio_file_path, zero_pad=True):
    """Return the time series of audio samples from the corresponding file"""

    # first read the raw binary file.
    binary = tf.io.read_file(audio_file_path)
    #decode binary into a 1D tensor containing the audio samples
    audio_tensor, _ = tf.audio.decode_wav(binary)
    audio_tensor = tf.squeeze(audio_tensor, axis=-1)
    if zero_pad:
        zero_padding = tf.zeros([16000] - tf.shape(audio_tensor), dtype=tf.float32)
        audio_tensor = tf.concat([audio_tensor, zero_padding], 0)
    return audio_tensor


# def get_mfcc_np(audio_file_path, winlen=0.025, winstep=0.01):
#     cc = mfcc(decode_audio(audio_file_path), samplerate=16000, winlen=winlen, winstep=winstep, numcep=13, nfilt=26, nfft=512, lowfreq=300, highfreq=8000)
#     deltas = delta(cc, 2)
#     accs = delta(deltas,2)
#     return tf.concat([cc, deltas, accs], axis=1)


if __name__ == "__main__":
    data_dir = pathlib.Path('data/speech_commands_v0.02')
    core_kws, aux_kws, output_classes = get_kws(data_dir)
    filenames0 = get_filenames(data_dir, core_kws + aux_kws)
    labels0 = np.array([get_label_int(file, aux_kws, output_classes) for file in filenames0])

    occs = []
    for i in range(19):
        occs.append(len(np.argwhere(labels0==i)))

    mean = int(np.mean(occs))
    #Balance filler words and other core keywords number

    i = np.argwhere(labels0==20)
    i = i.flatten()

    filler_indices = np.random.choice(i, mean, replace=False)

    filenames = []
    labels = []
    for i,name in enumerate(filenames0):
        if i not in filler_indices:
            filenames.append(name)
            labels.append(labels0[i])

    get_train_valid_test_split(filenames, labels, 0.8, 0.1, 0.1)
