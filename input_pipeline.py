import os
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
# from custom_layers import *
from hyperparams import _TASKS_,_DATA_DIR_, _BINARIES_DIR_, _NOISE_DIR_, _SILENCE_CLASS_, _UNKNOWN_CLASS_, CROP_WIDTH, SAMPLE_RATE, _TEST_DATA_DIR_

# # Set seed for experiment reproducibility
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

def get_noise_samples_names():
    noise_samples_names = tf.io.gfile.glob(str(_NOISE_DIR_)+'/*')
    noise_samples_names = [
        file for file in noise_samples_names if file.endswith('.wav')]
    return tf.constant(noise_samples_names)


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
        core_kws = commands.tolist()
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


def make_and_save_original_splits(task, return_canonical_test_set=True):
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

    X_train = list(set(filenames) - set(X_test) - set(X_valid))
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
        n_sil_train = np.unique(y_train, return_counts=True)[1][:10].mean(dtype=int)
        n_sil_valid = np.unique(y_valid, return_counts=True)[1][:10].mean(dtype=int)
        n_sil_test = np.unique(y_test, return_counts=True)[1][:10].mean(dtype=int)

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
    

    ## decide how many _unknown_ samples to include:
    if _UNKNOWN_CLASS_ in output_classes:
        
        n_unk_tr = np.unique(y_train, return_counts=True)[1][:10].mean(dtype=int)
        n_unk_val = np.unique(y_valid, return_counts=True)[1][:10].mean(dtype=int)
        n_unk_test = np.unique(y_test, return_counts=True)[1][:10].mean(dtype=int)

        indexes_tr = np.where(y_train==output_classes.index(_UNKNOWN_CLASS_))[0]
        tot_unkn_tr = len(indexes_tr)
        todel_tr = np.random.choice(indexes_tr, tot_unkn_tr - n_unk_tr, replace=False)

        indexes_val = np.where(y_valid==output_classes.index(_UNKNOWN_CLASS_))[0]
        tot_unkn_val = len(indexes_val)
        todel_val = np.random.choice(indexes_val, tot_unkn_val - n_unk_val, replace=False)

        indexes_test = np.where(y_test==output_classes.index(_UNKNOWN_CLASS_))[0]
        tot_unkn_test = len(indexes_test)
        todel_test = np.random.choice(indexes_test, tot_unkn_test - n_unk_test, replace=False)

        X_train = np.delete(X_train, todel_tr)
        y_train = np.delete(y_train, todel_tr)

        X_valid = np.delete(X_valid, todel_val)
        y_valid = np.delete(y_valid, todel_val)

        X_test = np.delete(X_test, todel_test)
        y_test = np.delete(y_test, todel_test)

    if return_canonical_test_set and task == "10kws+U+S":
        X_test, y_test = get_canonical_10kwstask_test_set()

    # shuffle
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


def load_original_splits(task, smoke_size=-1):
    """Load training/validation/test splits from disk"""
    X_train = np.load(_BINARIES_DIR_/f"{task}/X_train.npy")
    y_train = np.load(_BINARIES_DIR_/f"{task}/y_train.npy")
    X_valid = np.load(_BINARIES_DIR_/f"{task}/X_valid.npy")
    y_valid = np.load(_BINARIES_DIR_/f"{task}/y_valid.npy")
    X_test = np.load(_BINARIES_DIR_/f"{task}/X_test.npy")
    y_test = np.load(_BINARIES_DIR_/f"{task}/y_test.npy")

    X_train, y_train, X_valid, y_valid, X_test, y_test = get_smoke_sized(X_train, 
                                                                     y_train, 
                                                                     X_valid, 
                                                                     y_valid, 
                                                                     X_test, 
                                                                     y_test, 
                                                                     smoke_size=smoke_size)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_test_label_int(audio_file_path):
        data_dir = pathlib.Path(os.path.sep.join(
            audio_file_path.split(os.path.sep)[:2]))
        _, _, output_classes = get_kws(data_dir, "10kws+U+S")
        file_path_names = audio_file_path.split(os.path.sep)
        kw = file_path_names[2]
        if kw == '_silence_':
            return np.argmax(_SILENCE_CLASS_ == np.array(output_classes))
        elif kw == '_unknown_':
            return np.argmax(_UNKNOWN_CLASS_ == np.array(output_classes))
        return np.argmax(kw == np.array(output_classes))
def get_canonical_10kwstask_test_set():
    """Returns the test set provided by the creators of the Google Speech Commands dataset,
    used for easier reproducibility and comparability of results.
    It is available at http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"""

    X_test = get_filenames(_TEST_DATA_DIR_)
    y_test = [get_test_label_int(file) for file in X_test]

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_test, y_test



def decode_audio(audio_file_path, zero_pad=True):
    """Returns the time series of audio samples from the corresponding file"""

    # first read the raw binary file.
    binary = tf.io.read_file(audio_file_path)
    # decode binary into a 1D tensor containing the audio samples
    audio_tensor, _ = tf.audio.decode_wav(binary)
    audio_tensor = tf.squeeze(audio_tensor, axis=-1)
    if zero_pad:
        zero_padding = tf.zeros(
            [16000] - tf.shape(audio_tensor), dtype=tf.float32)
        audio_tensor = tf.concat([audio_tensor, zero_padding], 0)
    return audio_tensor


def randomly_crop_wave(waveform, crop_width=CROP_WIDTH):
    """Given an audio wave, returns a random crop of the wave of width `crop_width` samples"""
    if crop_width == -1:
        return waveform
    else:
        # add dimension to pass it to image.random_crop function
        waveform = tf.expand_dims(waveform, 0)
        waveform = tf.image.random_crop(value=waveform, size=(1, crop_width))
        # remove the extra dimension and normalize
        waveform = waveform[0, :] / tf.reduce_max(waveform[0, :])
        return waveform
        # return waveform[0]


def generate_noise_crop(sample_rate=16000, for_training=True, duration=1):
    n_samples = sample_rate*duration
    choice = tf.cast(tf.random.uniform([],0,len(get_noise_samples_names())), dtype=tf.int32)
    noise_file = tf.gather(get_noise_samples_names(), choice)
    w = decode_audio(noise_file, zero_pad=False)

    ##choose which half of the file from where to extract the random crop;
    # if we are generating noise for training, choose from the first half,
    # otherwise choose from the second half
    w_len = tf.shape(w)[0]
    if for_training:
        w = w[:w_len//2]
    else:
        w = w[w_len//2:]

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

def augment_sample(waveform):
    """Augments the current sample, only for training. The sample is randomly shifted left or right of a 
    quantity picked from U(-0.1,0.1).
    Input:
        - `waveform`: sample waveform; if it's a tensor of zeros, it means that it must return a random noise sample """
    if tf.reduce_sum(tf.cast(waveform == tf.zeros(SAMPLE_RATE), tf.int32))==SAMPLE_RATE:
        waveform =  generate_noise_crop(for_training=True)
    else:
        waveform = randomly_shift_waveform(waveform)
    return waveform

def gen_sample_valtest(waveform):
    if tf.reduce_sum(tf.cast(waveform == tf.zeros(SAMPLE_RATE), tf.int32))==SAMPLE_RATE:
        return generate_noise_crop(for_training=False)
    else:
        return waveform

def generate_sample(sample_name, preemph=0.97):
    """Generates audio wave from file name. Do not apply random preprocessing in this case.
    Input:
        - `sample_name`: name of the .wav file. if it's "silence", a tensor of zeros is generated.
        This is done in order to later convert those into random noise crops from the noise files"""
    if sample_name == _SILENCE_CLASS_:
        return tf.zeros(SAMPLE_RATE)
    w = decode_audio(sample_name)

    # apply pre-emphasis filter with coefficient = 0.97. 
    # Followed the implementation by: https://github.com/jameslyons/python_speech_features/blob/e280ac2b5797a3445c34820b0110885cd6609e5f/python_speech_features/sigproc.py#L133
    tf_signal = tf.expand_dims(w,0)
    tf_emph = tf.concat([tf_signal[:,0:1], tf_signal[:,1:] - preemph * tf_signal[:,:-1]], axis=1)

    # normalize so that every waveform is between 1 and -1
    audio_tensor = tf_emph[0] / tf.reduce_max(tf_emph[0])
    return audio_tensor
    # return tf_emph[0]

def create_dataset(files_ds,
                   labels,
                   batch_size,
                   is_training=False,
                   is_test=False,
                   cache_file=None):
    dataset = tf.data.Dataset.from_tensor_slices((files_ds, labels))
    dataset = dataset.map(lambda file_name, label: (generate_sample(file_name), label), num_parallel_calls=tf.data.AUTOTUNE)

    # will cache only training set
    if cache_file and is_training:
        dataset = dataset.cache(cache_file)
    
    if not is_training: # i.e. only for test and validation
        #convert tensors of zeros to random noise crops which will be cached
        dataset = dataset.map(lambda file_name, label: (gen_sample_valtest(file_name), label), num_parallel_calls=tf.data.AUTOTUNE)
        if cache_file:
            dataset = dataset.cache(cache_file)

    #apply repeat only on training and validation:
    if not is_test:
        dataset = dataset.repeat()

    # After caching, apply data augmentation (random shift and generation of random silence samples), 
    # ONLY to training samples
    if is_training:
        dataset = dataset.map(lambda waveform, label: (augment_sample(waveform), label), num_parallel_calls=tf.data.AUTOTUNE)

    if not is_test:
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
        
def get_tf_datasets(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, task, verbose=True):     

    train_dataset = create_dataset(X_train,
                                y_train, 
                                batch_size=batch_size, 
                                is_training=True,
                                cache_file=f'cache_training_{task}')
                                # cache_file=None)

    valid_dataset = create_dataset(X_valid,
                                y_valid,
                                batch_size=batch_size, 
                                cache_file=f'cache_valid_{task}')
                                # cache_file=None)

    test_dataset = create_dataset(X_test,
                                y_test,
                                batch_size=batch_size, 
                                is_test=True,
                                cache_file=f'cache_test_{task}')

    train_steps = int(np.ceil(len(X_train)/batch_size))
    valid_steps = int(np.ceil(len(X_valid)/batch_size))
    # test_steps = int(np.ceil(len(X_test)/batch_size))
    if verbose:
        print(f"Train steps: {train_steps}")
        print(f"Validations steps: {valid_steps}")
        # print(f"Test steps: {test_steps}")

        for i in train_dataset.take(1):
            print("Example of dataset element:")
            print(i)
    
    return train_dataset, train_steps, valid_dataset, valid_steps, test_dataset

def get_noises_tf_dataset():
    names = get_noise_samples_names().numpy()
    names = [n.decode() for n in names]
    noises_ds = tf.data.Dataset.from_tensor_slices(names)
    noises_ds = noises_ds.map(lambda f: decode_audio(f, zero_pad=False))
    noises_ds = noises_ds.map(lambda x: x[:800000])
    noises_ds = noises_ds.batch(6).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return noises_ds


if __name__ == "__main__":
    ex = 'data/speech_commands_v0.02/up/c7aaad67_nohash_2.wav'
    generate_noise_crop()