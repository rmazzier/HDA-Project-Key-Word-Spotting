import os
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from custom_layers import *
from hyperparams import _TASKS_,_DATA_DIR_, _BINARIES_DIR_, _NOISE_DIR_, _SILENCE_CLASS_, _UNKNOWN_CLASS_

def get_noise_samples_names():
    noise_samples_names = tf.io.gfile.glob(str(_NOISE_DIR_)+'/*')
    noise_samples_names = [
        file for file in noise_samples_names if file.endswith('.wav')]
    return noise_samples_names


def get_kws(data_dir, task):
    """Scans the data directory and outputs the core keywords (the ones that our NN will have to learn to classify),
    the auxiliary keywords (the ones to be classified in the "filler" class), and the list of actual output classes
    for our model."""
    assert(task in _TASKS_)

    commands = np.array(tf.io.gfile.listdir(data_dir))
    commands = commands[commands != 'README.md']
    commands = commands[commands != 'LICENSE']
    commands = commands[commands != 'testing_list.txt']
    commands = commands[commands != 'validation_list.txt']
    commands = commands[commands != '_background_noise_']
    commands = commands[commands != '.DS_Store']

    if task == "10kws+U+S":
        core_kws = ["yes", "no", "up", "down", "left",
                    "right", "on", "off", "stop", "go"]
        aux_kws = list(set(commands) - set(core_kws))
        output_classes = core_kws+[_UNKNOWN_CLASS_, _SILENCE_CLASS_]
    elif task == "20kws+U":
        core_kws = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop",
                    "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        aux_kws = list(set(commands) - set(core_kws))
        output_classes = core_kws+[_UNKNOWN_CLASS_]

    else:
        core_kws = commands
        aux_kws = []
        output_classes = core_kws
    
    return core_kws, aux_kws, output_classes


def get_filenames(data_dir):
    """Scans each directory inside the data directory for all the audio files. 
    Returns a list with all the audio files names"""
    core_kws, aux_kws, _ = get_kws(data_dir, _TASKS_[0])
    commands = core_kws + aux_kws
    filenames = []
    for command in commands:
        filenames += tf.io.gfile.glob(str(data_dir) +
                                      os.path.sep + command + '/*')
    return np.array(filenames)


def add_silence_samples(X_train, y_train, X_valid, y_valid, X_test, y_test, amount_train, amount_valid, amount_test, task):
    """From a starting train/valid/test dataset of file names, add 'silence' entries. Those will be converted to random crops of noise waves.
    During the creation of the tf.Dataset. The corresponding label will be the index of the _SILENCE_CLASS_ entry in the output_classes vector."""
    _, _, output_classes = get_kws(_DATA_DIR_, task)
    sil_index = output_classes.index(_SILENCE_CLASS_)
    X_train_s = np.concatenate(
        (X_train, [_SILENCE_CLASS_ for i in range(amount_train)]))
    y_train_s = np.concatenate(
        (y_train, [sil_index for i in range(amount_train)]))

    X_valid_s = np.concatenate(
        (X_valid, [_SILENCE_CLASS_ for i in range(amount_valid)]))
    y_valid_s = np.concatenate(
        (y_valid, [sil_index for i in range(amount_valid)]))

    X_test_s = np.concatenate(
        (X_test, [_SILENCE_CLASS_ for i in range(amount_test)]))
    y_test_s = np.concatenate(
        (y_test, [sil_index for i in range(amount_test)]))

    # shuffle
    train_indexes = np.random.permutation(range(len(X_train_s)))
    valid_indexes = np.random.permutation(range(len(X_valid_s)))
    test_indexes = np.random.permutation(range(len(X_test_s)))

    return X_train_s[train_indexes], y_train_s[train_indexes], X_valid_s[valid_indexes], y_valid_s[valid_indexes], X_test_s[test_indexes], y_test_s[test_indexes]


def get_label_int(audio_file_path, task):
    """Input: path of the audio file;
       Output: integer containing index of keyword associated to the audio file. """
    assert(task in _TASKS_)
    data_dir = pathlib.Path(os.path.sep.join(
        audio_file_path.split(os.path.sep)[:2]))
    _, aux_kws, output_classes = get_kws(data_dir, task)

    file_path_names = audio_file_path.split(os.path.sep)
    kw = file_path_names[2]
    if kw in aux_kws:
        return np.argmax(_UNKNOWN_CLASS_ == np.array(output_classes))
    else:
        return np.argmax(kw == np.array(output_classes))


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


def get_original_splits(task):
    """Get the Train Validation Test splits provided by the dataset."""
    assert(task in _TASKS_)
    filenames = get_filenames(_DATA_DIR_)

    test_names_file = _DATA_DIR_/"testing_list.txt"
    valid_names_file = _DATA_DIR_/"validation_list.txt"

    X_test = []
    y_test = []
    X_valid = []
    y_valid = []

    f = open(test_names_file, "r")
    for file in f:
        file = str(_DATA_DIR_/file)
        X_test.append(file[:-1])
        y_test.append(get_label_int(audio_file_path=file, task=task))
    f.close()

    f = open(valid_names_file, "r")
    for file in f:
        file = str(_DATA_DIR_/file)

        X_valid.append(file[:-1])
        y_valid.append(get_label_int(audio_file_path=file, task=task))
    f.close()

    X_train = list(set(filenames) - set(X_test))
    y_train = [get_label_int(file, task=task) for file in X_train]

    _,_,output_classes = get_kws(_DATA_DIR_, task)

    
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)

    #IF the current task has "silence" as one possible output class, add the silence samples
    if _SILENCE_CLASS_ in output_classes:
        print("Adding Silence Samples....")
        n_sil_train = len(y_train[y_train==0])
        n_sil_valid = len(y_valid[y_valid==0])
        n_sil_test = len(y_test[y_test==0])
        X_train, y_train, X_valid, y_valid, X_test, y_test = add_silence_samples(X_train, 
                                                                                y_train, 
                                                                                X_valid, 
                                                                                y_valid, 
                                                                                X_test, 
                                                                                y_test,
                                                                                n_sil_train,
                                                                                n_sil_valid,
                                                                                n_sil_test,
                                                                                task)
    # shuffle
    print("Shuffling dataset...")
    train_indexes = np.random.permutation(range(len(X_train)))
    valid_indexes = np.random.permutation(range(len(X_valid)))
    test_indexes = np.random.permutation(range(len(X_test)))

    X_train = X_train[train_indexes]
    y_train = y_train[train_indexes]
    X_valid = X_valid[valid_indexes]
    y_valid = y_valid[valid_indexes]
    X_test = X_test[test_indexes]
    y_test = y_test[test_indexes]


    np.save(_BINARIES_DIR_/f"{task}/X_train", X_train)
    np.save(_BINARIES_DIR_/f"{task}/y_train", y_train)
    np.save(_BINARIES_DIR_/f"{task}/X_test", X_test)
    np.save(_BINARIES_DIR_/f"{task}/y_test", y_test)
    np.save(_BINARIES_DIR_/f"{task}/X_valid", X_valid)
    np.save(_BINARIES_DIR_/f"{task}/y_valid", y_valid)

    return X_train, y_train, X_valid, y_valid,X_test, y_test


def load_original_splits(task, smoke_size=-1):
    """Load training/validation/test splits from disk"""
    X_train = np.load(_BINARIES_DIR_/f"{task}/X_train.npy")
    y_train = np.load(_BINARIES_DIR_/f"{task}/y_train.npy")
    X_valid = np.load(_BINARIES_DIR_/f"{task}/X_test.npy")
    y_valid = np.load(_BINARIES_DIR_/f"{task}/y_test.npy")
    X_test = np.load(_BINARIES_DIR_/f"{task}/X_valid.npy")
    y_test = np.load(_BINARIES_DIR_/f"{task}/y_valid.npy")

    X_train, y_train, X_valid, y_valid, X_test, y_test = get_smoke_sized(X_train, 
                                                                     y_train, 
                                                                     X_valid, 
                                                                     y_valid, 
                                                                     X_test, 
                                                                     y_test, 
                                                                     smoke_size=smoke_size)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def decode_audio(audio_file_path, zero_pad=True):
    """Returns the time series of audio samples from the corresponding file"""

    # first read the raw binary file.
    binary = tf.io.read_file(audio_file_path)
    # decode binary into a 1D tensor containing the audio samples
    audio_tensor, _ = tf.audio.decode_wav(binary)
    audio_tensor = tf.squeeze(audio_tensor, axis=-1)
    # normalize so that every waveform is between 1 and -1
    audio_tensor = audio_tensor / tf.reduce_max(audio_tensor)
    if zero_pad:
        zero_padding = tf.zeros(
            [16000] - tf.shape(audio_tensor), dtype=tf.float32)
        audio_tensor = tf.concat([audio_tensor, zero_padding], 0)
    return audio_tensor


# def randomly_crop_wave(waveform, crop_width=_CROP_WIDTH_):
#     """Given an audio wave, returns a random crop of the wave of width `crop_width` samples"""
#     if crop_width == -1:
#         return waveform
#     else:
#         # add dimension to pass it to image.random_crop function
#         waveform = tf.expand_dims(waveform, 0)
#         waveform = tf.image.random_crop(value=waveform, size=(1, crop_width))
#         # remove the extra dimension and normalize
#         waveform = waveform[0, :] / tf.reduce_max(waveform[0, :])
#         return waveform


def generate_noise_crop(sample_rate=16000, duration=1):
    n_samples = sample_rate*duration
    noise_file = np.random.choice(get_noise_samples_names())
    w = decode_audio(noise_file, zero_pad=False)
    w = randomly_crop_wave(w, n_samples)
    return w

def randomly_shift_waveform(waveform):
    """Given a waveform, it randomly shifts it left or right at most by 0.1 seconds.
    The remaining part gets zero padded"""
    amt = tf.random.uniform(shape=[], minval=-0.1, maxval=0.1)
    amt_samples = tf.cast(tf.math.round(16000 * amt), dtype=tf.int32)

    new_w = tf.roll(waveform, amt_samples, axis=0)
    zeros = tf.zeros(tf.abs(amt_samples))
    
    if amt_samples > 0:
        new_w = new_w[amt_samples:]
        new_w = tf.concat([zeros, new_w], axis=0)
    elif amt_samples < 0:
        new_w = new_w[:amt_samples]
        new_w = tf.concat([new_w, zeros], axis=0)
    else:
        pass

    return new_w

def generate_train_sample(sample_name, noise_prob=0.8, preemph=0.97):
    """Generates audio wave from file name, and with probability p the sample is mixed with noise,
    and randomly shifted left or right of a quantity picked from U(-0.1,0.1)
    Input:
        - `sample_name`: name of the .wav file. if it's "silence", a noise sample is generated;"""
    if sample_name == "silence":
        return generate_noise_crop()
    w = decode_audio(sample_name)
    if tf.random.uniform(shape=[], minval=0., maxval=1.) <= noise_prob:
        coeff = tf.random.uniform(shape=[], minval=0., maxval=0.1)
        noise = generate_noise_crop() * coeff
        w = noise + w
    w = randomly_shift_waveform(w)
    # apply pre-emphasis filter with coefficient = 0.97. 
    # Followed the implementation by: https://github.com/jameslyons/python_speech_features/blob/e280ac2b5797a3445c34820b0110885cd6609e5f/python_speech_features/sigproc.py#L133
    tf_signal = tf.expand_dims(w,0)
    tf_emph = tf.concat([tf_signal[:,0:1], tf_signal[:,1:] - preemph * tf_signal[:,:-1]], axis=1)
    return tf_emph[0]

def generate_test_sample(sample_name, preemph=0.97):
    """Generates audio wave from file name. Do not apply random preprocessing in this case.
    Input:
        - `sample_name`: name of the .wav file. if it's "silence", a noise sample is generated;"""
    if sample_name == "silence":
        return generate_noise_crop()
    w = decode_audio(sample_name)

    # apply pre-emphasis filter with coefficient = 0.97. 
    # Followed the implementation by: https://github.com/jameslyons/python_speech_features/blob/e280ac2b5797a3445c34820b0110885cd6609e5f/python_speech_features/sigproc.py#L133
    tf_signal = tf.expand_dims(w,0)
    tf_emph = tf.concat([tf_signal[:,0:1], tf_signal[:,1:] - preemph * tf_signal[:,:-1]], axis=1)
    return tf_emph[0]

def create_dataset(files_ds,
                   labels,
                   batch_size,
                   shuffle=False,
                   cache_file=None,
                   is_test=False):
    dataset = tf.data.Dataset.from_tensor_slices((files_ds, labels))

    if is_test:
        dataset = dataset.map(lambda file_name, label: (generate_test_sample(file_name), label), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda file_name, label: (generate_train_sample(file_name), label), num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle dataset
    if shuffle:
        dataset = dataset.shuffle(len(files_ds))
    
    # Repeat dataset indefinitely (useful when len(dataset)/batch_size is not integer. In those cases, reuse data
    # in order to have batches of equal dimension)
    # When I am generating the dataset, I always want to generate that same set in order to have consistent
    # comparisons between models, so I don't call repeat in that case.
    if not is_test:
        # # Cache dataset
        if cache_file:
            dataset = dataset.cache(cache_file)
        dataset = dataset.repeat()

    # Batch
    dataset = dataset.batch(batch_size=batch_size)

    # Prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def get_smoke_sized(X_train, y_train, X_valid, y_valid, X_test, y_test, smoke_size=0):
    if smoke_size > 0:
        names_train = X_train[:smoke_size]
        labels_train = y_train[:smoke_size]
        names_valid = X_valid[:round(smoke_size/10)]
        labels_valid = y_valid[:round(smoke_size/10)]
        names_test = X_test[:round(smoke_size/10)]
        labels_test= y_test[:round(smoke_size/10)]
    else:
        names_train = X_train
        labels_train = y_train
        names_valid = X_valid
        labels_valid = y_valid
        names_test = X_test
        labels_test= y_test
    
    return names_train, labels_train, names_valid, labels_valid, names_test, labels_test
        
def get_tf_datasets(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, task, shuffle=False):     

    train_dataset = create_dataset(X_train,
                                y_train, 
                                batch_size=batch_size, 
                                shuffle=shuffle,
                                cache_file=f'cache_training_{task}')

    valid_dataset = create_dataset(X_valid,
                                y_valid,
                                batch_size=batch_size, 
                                shuffle=shuffle,
                                cache_file=f'cache_valid_{task}')

    test_dataset = create_dataset(X_test,
                                y_test,
                                batch_size=batch_size, 
                                shuffle=shuffle,
                                is_test=True)

    train_steps = int(np.ceil(len(X_train)/batch_size))
    valid_steps = int(np.ceil(len(X_valid)/batch_size))

    print(f"Train steps: {train_steps}")
    print(f"Validations steps: {valid_steps}")

    for i in train_dataset.take(1):
        print("Example of dataset element:")
        print(i)
    
    return train_dataset, train_steps, valid_dataset, valid_steps, test_dataset



# if __name__ == "__main__":
#     X_train, y_train, X_valid, y_valid, X_test, y_test = get_original_splits(_DATA_DIR_)