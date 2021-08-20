import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.initializers import glorot_uniform
import math

class Spectrogram(tf.keras.layers.Layer):
    """Compute spectrogram from waveform."""

    def __init__(self, sample_rate, fft_size, win_size, hop_size,
                 f_min=0.0, f_max=None, **kwargs):
        super(Spectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.win_size = win_size
        self.hop_size = hop_size

    def build(self, input_shape):
        super(Spectrogram, self).build(input_shape)

    def call(self, waveforms):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (batch_size, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        spectrograms : (tf.Tensor), shape = (None, audio_frames, fft_size/2 + 1, ch)
            The corresponding batch of spectrograms.
        """

        # compute spectrogram with STFT; shape: (batch_size, n_frames, fft_size/2 +1)
        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.win_size,
                                      frame_step=self.hop_size,
                                      fft_length=self.fft_size)
        # get absolute value 
        spectrograms = tf.abs(spectrograms)
        return tf.expand_dims(spectrograms, -1)

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'win_size': self.win_size,
            'sample_rate': self.sample_rate
        }
        config.update(super(Spectrogram, self).get_config())

        return config

class LogMelSpectrogram(tf.keras.layers.Layer):
    """Compute log_mel_spectrogram from waveform."""

    def __init__(self, sample_rate, fft_size, win_size, hop_size, n_filters,
                 f_min=0.0, f_max=None, **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.win_size = win_size
        self.hop_size = hop_size
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_filters,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max)

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LogMelSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (batch_size, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        log_mel_spectrograms : (tf.Tensor), shape = (None, audio_frames, n_filters, ch)
            The corresponding batch of log-mel-spectrograms
        """

        # compute spectrogram with STFT; shape: (batch_size, n_frames, fft_size/2 +1)
        # spectrograms = tf.signal.stft(waveforms,
        #                               frame_length=self.win_size,
        #                               frame_step=self.hop_size,
        #                               fft_length=self.fft_size)
        # # get absolute value 
        # spectrograms = tf.abs(spectrograms)

        spectrograms = Spectrogram(sample_rate=self.sample_rate, 
            fft_size=self.fft_size, 
            win_size=self.win_size, 
            hop_size=self.hop_size,
            f_min=self.f_min,
            f_max=self.f_max)(waveforms)[...,0] #remove last dimension 

        # for some reason this passage gives a strange result...
        #spectrogram=tf.multiply(tf.math.square(spectrogram), 1/_FFT_SIZE)

        #compute energy for each frame; energy will be of shape (batch_size, n_frames)
        energy = tf.reduce_sum(tf.multiply(tf.math.square(spectrograms), 1/self.fft_size), axis=2)
        energy = tf.expand_dims(energy, -1)

        # map from linear frequency scale to mel scale

        # linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        #     num_mel_bins=self.n_filters,
        #     num_spectrogram_bins=self.fft_size//2 + 1,
        #     sample_rate=self.sample_rate,
        #     lower_edge_hertz=self.f_min,
        #     upper_edge_hertz=self.f_max)

        mel_spectrograms = tf.tensordot(
            spectrograms, self.mel_filterbank, 1)

        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        # concatenate vector of energies to the log_spectrogram
        log_mel_spectrograms_e = tf.concat([energy, log_mel_spectrograms],2)
        return tf.expand_dims(log_mel_spectrograms_e, -1)

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'win_size': self.win_size,
            'n_filters': self.n_filters,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(LogMelSpectrogram, self).get_config())

        return config

class MFCC(tf.keras.layers.Layer):
    """Compute mfcc from waveform.
    if `return_deltas` is true, delta MFCCs will be stacked on top of the regular MFCCs."""

    def __init__(self, sample_rate, fft_size, win_size, hop_size, n_filters, n_cepstral,
                 f_min=0.0, f_max=None, return_deltas=True, **kwargs):
        super(MFCC, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.win_size = win_size
        self.hop_size = hop_size
        self.n_filters = n_filters
        self.n_cepstral = n_cepstral
        self.return_deltas = return_deltas
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2

    def build(self, input_shape):
        self.non_trainable_weights.append(self.sample_rate)
        self.non_trainable_weights.append(self.fft_size)
        self.non_trainable_weights.append(self.win_size)
        self.non_trainable_weights.append(self.hop_size)
        self.non_trainable_weights.append(self.n_filters)
        self.non_trainable_weights.append(self.n_cepstral)
        self.non_trainable_weights.append(self.return_deltas)
        self.non_trainable_weights.append(self.f_min)
        self.non_trainable_weights.append(self.f_max)
        super(MFCC, self).build(input_shape)

    def delta_tf(self, features, N):
        """Tensor of features of shape (batch_size, n_time_frames, n_cepstra)
        Parameter N: For each frame, calculate delta features based on preceding and following N frames"""
        NUMFRAMES = features.shape[1]
        denominator = tf.reduce_sum(
            tf.square(tf.range(1., N+1.)))*2.

        # compute padded tensor of features
        l1 = tf.expand_dims(features[:, 0, :], 1)
        ln = tf.expand_dims(features[:, -1, :], 1)
        l1 = tf.repeat(l1, N, axis=1)
        ln = tf.repeat(ln, N, axis=1)
        padded_tf = tf.concat([l1, features, ln], axis=1)

        def delta_t(t):
            r = tf.range(-N, N+1, dtype=tf.float32)
            r = tf.expand_dims(r, 0)
            r = tf.expand_dims(r, -1)
            sl = padded_tf[:, t: t+2*N+1, :]
            ss = tf.multiply(r, sl)
            return tf.reduce_sum(ss, 1)

        delta_0 = tf.map_fn(fn=lambda t: delta_t(t), elems=tf.range(NUMFRAMES), parallel_iterations=10, fn_output_signature=tf.float32) / denominator
        return tf.transpose(delta_0, perm=(1, 0, 2))

    def tf_lift(self, mfccs, L=22):
        """Applies liftering to the mfccs matrix."""
        n = tf.range(mfccs.shape[2], dtype=tf.float32)
        lift = 1.0 + (L/2.0)*tf.math.sin(math.pi*n/L)
        return mfccs * lift

    def call(self, waveforms):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        mfccs : (tf.Tensor), shape = (None, audio_frames, n_cepstral, ch)
            The corresponding batch of mfccs
        """

        # Remember: first column o each sample is the vector of energies for each frame
        log_mel_spectrograms = LogMelSpectrogram(sample_rate=self.sample_rate, fft_size=self.fft_size,
                                              win_size=self.win_size, hop_size=self.hop_size, n_filters=self.n_filters)(waveforms)
                                            
        log_energies = tf.math.log(log_mel_spectrograms[:,:,0,:] + 1e-6)

        # Now compute MFCCs from log-magnitude mel scaled spectrogram
        # NB: from TF documentation:
        # input is a [..., num_mel_bins] float32/float64 Tensor of log-magnitude mel-scale spectrograms.
        # Since log_mel_spectrograms comes from custom layer, we have to remove the last dimension.
        # I also don't take the first column since it's the vector of energies

        # I then take the Cepstral Coefficients from the 2nd to the n_cepstral-th. Later will stack the vector of energies
        # in place of the first CC.
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrograms[:,:,1:, 0])[..., 1:self.n_cepstral]
        mfccs = self.tf_lift(mfccs)

        mfccs = tf.concat([log_energies, mfccs], 2)
        if self.return_deltas:
            mfccs = tf.concat([mfccs, self.delta_tf(mfccs,2), self.delta_tf(self.delta_tf(mfccs,2), 2) ], axis=2)
        return tf.expand_dims(mfccs, -1)

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'win_size': self.win_size,
            'n_filters': self.n_filters,
            'n_cepstral': self.n_cepstral,
            'return_deltas': self.return_deltas,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(MFCC, self).get_config())

        return config

## RESNET BLOCKS ###

def identity_block(X_input, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters

    # First component of main path
    X = layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), 
               padding = 'valid', name = conv_name_base + '1th', 
               kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '1th')(X)
    X = layers.Activation('relu')(X)
    
    # Second component of main path
    X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), 
               padding = 'same', name = conv_name_base + '2nd', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis =-1, name = bn_name_base + '2nd')(X)
    X = layers.Activation('relu')(X)

    # Third component of main path
    X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), 
               padding = 'valid', name = conv_name_base + '3rd', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '3rd')(X)
    X = layers.Add()([X_input, X])
    X = layers.Activation('relu')(X)    
    return X

def convolutional_block(X_input, f, filters, stage, block, s = 2):
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters

    ##### MAIN PATH ##### 
    # First component of main path
    X = layers.Conv2D(F1, (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + '1st', 
               kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '1st')(X)
    X = layers.Activation('relu')(X)
    
    # Second component of main path
    X = layers.Conv2D(F2, (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2nd', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '2nd')(X)
    X = layers.Activation('relu')(X)

    # Third component of main path 
    X = layers.Conv2D(F3, (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '3rd', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '3rd')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = layers.Conv2D(F3, (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + '1', 
               kernel_initializer = glorot_uniform(seed=0))(X_input)
    X_shortcut = layers.BatchNormalization(axis = -1, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X_shortcut, X])
    X = layers.Activation('relu')(X)

    
    return X
