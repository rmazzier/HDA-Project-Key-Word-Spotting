import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform
import tensorflow_addons as tfa
import math
import hyperparams
from input_pipeline import get_noises_tf_dataset


class RandomNoiseAugment(tf.keras.layers.Layer):
    def __init__(self, sample_rate=16000, **kwargs):
        super(RandomNoiseAugment, self).__init__(**kwargs)
        # self.ds_noise = ds_noise
        self.ds_noise = get_noises_tf_dataset()
        self.sample_rate = sample_rate
        for i in iter(self.ds_noise):
            #single batch containing all the noise files names
            self.noises = i
        #normalize
        self.noises = self.noises / tf.expand_dims(tf.reduce_max(self.noises, axis=1),1)

    def build(self, input_shape):
        super(RandomNoiseAugment, self).build(input_shape)
    
    @tf.function
    def choose(self,i,noises):
        return noises[i]

    def call(self, waveforms, training):
        if training:
            b_size = tf.shape(waveforms)[0]

            ## TODO: Right now, i am adding noise over noise... this can be harmful?? or not?

            ## Randomly translate the batch of noises before cropping?
            # this would misalign the noise files so that when making a random 16000 crop
            # it would be like taking random 1 second crops from the files
            
            #crop 1second width
            noises = tf.image.random_crop(self.noises, size=(6,self.sample_rate))
            
            # tensor of indexes for noise waves
            z = tf.cast(tf.random.uniform([b_size],0,6), dtype=tf.int32)
        
            ## batch of random noises of shape [batch_size, sample_rate]
            noises_batch = tf.vectorized_map(lambda i : self.choose(i,noises), z)
            
            # each sample is mixed with noise with probability 0.8
            ps = tf.expand_dims(tf.cast(tf.random.uniform([b_size])<0.8, tf.float32),-1)
            noises_batch = ps * noises_batch * tf.random.uniform(shape=[], minval=0., maxval=0.2)
            return waveforms + noises_batch
        else:
            return waveforms

    def get_config(self):
        config = {
            'ds_noise': self.ds_noise,
            'noises':self.ds_noise,
            'sample_rate': self.sample_rate
        }
        config.update(super(RandomNoiseAugment, self).get_config())

        return config


class Spectrogram(tf.keras.layers.Layer):
    """Compute spectrogram from waveform."""

    def __init__(self, 
        sample_rate=hyperparams.SAMPLE_RATE, 
        fft_size=hyperparams.FFT_SIZE, 
        win_size=hyperparams.WIN_SIZE, 
        hop_size=hyperparams.HOP_SIZE, **kwargs):

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

class SpecAugment(tf.keras.layers.Layer):
    """Custom layer that applies data aumentation according to the SpecAugment policy from: https://arxiv.org/pdf/1904.08779.pdf """
    def __init__(self, F=hyperparams.F, T=hyperparams.T, **kwargs):
        super(SpecAugment, self).__init__(**kwargs)
        self.F = F
        self.T = T

    def build(self, input_shape):
        self.non_trainable_weights.append(self.F)
        self.non_trainable_weights.append(self.T)
        super(SpecAugment, self).build(input_shape)
    
    @tf.function
    def cutout_single(self, i, inputs,t,f):
            X = inputs[i,...]
            ti = t[i]
            fi = f[i]
            # if tf.random.uniform([]) < self.p:
            # f = tf.random.uniform([], minval=0, maxval=self.F, dtype=tf.int32)
            # t = tf.random.uniform([], minval=0, maxval=self.T, dtype=tf.int32)
            # X = tf.expand_dims(X,0)
            n_channels = X.shape[1]
            n_time_steps = X.shape[0]
            f0 =  tf.random.uniform([], minval=0, maxval=n_channels-fi, dtype=tf.int32)
            t0 =  tf.random.uniform([],minval=0, maxval=n_time_steps-ti, dtype=tf.int32)
            # apply masks in the time and frequency domains
            # X2 = tfa.image.random_cutout(X, (t, n_channels+2))
            # X3 = tfa.image.random_cutout(X2, (n_time_steps+2, f))

            ## time mask
            ones1t = tf.ones((n_channels, t0,1))
            zerost = tf.zeros((n_channels, ti,1))
            ones2t = tf.ones((n_channels, n_time_steps - t0 -ti,1))
            tmask = tf.concat([ones1t, zerost, ones2t], axis=1)
            tmask = tf.transpose(tmask, [1,0,2])
            # tmask = tf.expand_dims(tmask,-1)b

            ## frequency mask
            ones1f = tf.ones((f0, n_time_steps,1))
            zerosf = tf.zeros((fi, n_time_steps,1))
            ones2f = tf.ones((n_channels - f0 -fi, n_time_steps,1))
            fmask = tf.concat([ones1f, zerosf, ones2f], axis=0)
            fmask = tf.transpose(fmask, [1,0,2])
            # fmask = tf.expand_dims(fmask,-1)

            # multiply mask
            X_masked = X * tmask * fmask
            return X_masked
            # else:
            #     return X

    def call(self, inputs, training):
        if training:
            f = tf.random.uniform([tf.shape(inputs)[0]], minval=0, maxval=self.F, dtype=tf.int32)
            t = tf.random.uniform([tf.shape(inputs)[0]], minval=0, maxval=self.T, dtype=tf.int32)

            X4 = tf.map_fn(lambda i: self.cutout_single(i, inputs,t,f), tf.range(tf.shape(inputs)[0]), parallel_iterations=500, fn_output_signature=tf.float32)

            return X4
        else:
            return inputs

    def get_config(self):
        config = {
            'F': self.F,
            'T' : self.T
        }
        config.update(super(SpecAugment, self).get_config())
        return config

class LogMelSpectrogram(tf.keras.layers.Layer):
    """Compute log_mel_spectrogram from waveform."""

    def __init__(self, sample_rate = hyperparams.SAMPLE_RATE, 
        fft_size=hyperparams.FFT_SIZE, 
        win_size=hyperparams.WIN_SIZE, 
        hop_size=hyperparams.HOP_SIZE, 
        n_filters=hyperparams.N_FILTERS,
        f_min=300.0, 
        f_max=None, **kwargs):

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
                                   hop_size=self.hop_size)(waveforms)[..., 0]  # remove last dimension

        # compute energy for each frame; energy will be of shape (batch_size, n_frames)
        energy = tf.reduce_sum(tf.multiply(
            tf.math.square(spectrograms), 1/self.fft_size), axis=2)
        energy = tf.expand_dims(energy, -1)

        mel_spectrograms = tf.tensordot(
            spectrograms, self.mel_filterbank, 1)

        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        # concatenate vector of energies to the log_spectrogram
        log_mel_spectrograms_e = tf.concat([energy, log_mel_spectrograms], 2)
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

    def __init__(self, 
        sample_rate = hyperparams.SAMPLE_RATE, 
        fft_size = hyperparams.FFT_SIZE, 
        win_size = hyperparams.WIN_SIZE, 
        hop_size = hyperparams.HOP_SIZE,
        n_filters = hyperparams.N_FILTERS, 
        n_cepstral = hyperparams.N_CEPSTRAL,
        lift_constant = hyperparams.L,
        f_min=300.0, 
        f_max=None, 
        return_deltas=hyperparams.DELTAS, **kwargs):

        super(MFCC, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.win_size = win_size
        self.hop_size = hop_size
        self.n_filters = n_filters
        self.n_cepstral = n_cepstral
        self.lift_constant = lift_constant
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
        self.non_trainable_weights.append(self.lift_constant)
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

        delta_0 = tf.map_fn(fn=lambda t: delta_t(t), elems=tf.range(
            NUMFRAMES), parallel_iterations=10, fn_output_signature=tf.float32) / denominator
        return tf.transpose(delta_0, perm=(1, 0, 2))

    def tf_lift(self, mfccs):
        """Applies liftering to the mfccs matrix."""
        n = tf.range(mfccs.shape[2], dtype=tf.float32)
        lift = 1.0 + (self.lift_constant/2.0)*tf.math.sin(math.pi*n/self.lift_constant)
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

        log_energies = tf.math.log(log_mel_spectrograms[:, :, 0, :] + 1e-6)

        # Now compute MFCCs from log-magnitude mel scaled spectrogram
        # NB: from TF documentation:
        # input is a [..., num_mel_bins] float32/float64 Tensor of log-magnitude mel-scale spectrograms.
        # Since log_mel_spectrograms comes from custom layer, we have to remove the last dimension.
        # I also don't take the first column since it's the vector of energies

        # I then take the Cepstral Coefficients from the 2nd to the n_cepstral-th. Later will stack the vector of energies
        # in place of the first CC.
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrograms[:, :, 1:, 0])[..., 1:self.n_cepstral]
        mfccs = self.tf_lift(mfccs)

        mfccs = tf.concat([log_energies, mfccs], 2)
        if self.return_deltas:
            mfccs = tf.concat([mfccs, self.delta_tf(mfccs, 2), self.delta_tf(
                self.delta_tf(mfccs, 2), 2)], axis=2)
        return tf.expand_dims(mfccs, -1)

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'win_size': self.win_size,
            'n_filters': self.n_filters,
            'n_cepstral': self.n_cepstral,
            'lift_constant': self.lift_constant,
            'return_deltas': self.return_deltas,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(MFCC, self).get_config())

        return config
    
class PosAndClassEmbed(tf.keras.layers.Layer):
    def __init__(self, num_patches, d_model, **kwargs):
        super(PosAndClassEmbed, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.d_model = d_model
        
    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            "pos_emb", 
            shape=(1, self.num_patches + 1, self.d_model), 
            initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02))

        self.class_emb = self.add_weight(
            "class_emb", 
            shape=(1, 1, self.d_model), 
            initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02))
        
    def call(self, x):
        batch_size = tf.shape(x)[0]

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
 
        tokens = [class_emb, x]

        x = tf.concat(tokens, axis=1)
        return x + self.pos_emb
    
    def get_config(self):
        config = {
            'pos_emb' : self.pos_emb,
            'class_emb' : self.class_emb
        }
        config.update(super(PosAndClassEmbed, self).get_config())

        return config


class TransformerBlock(tf.keras.layers.Layer):
    """Implementation of the Transformer Block
        - embed_dim: dimension of the embedding for the Multi Head Attention layer;
        - num_heads: number of heads for the Multi Head Attention layer"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation=tf.keras.activations.gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=0., stddev=0.02), bias_initializer=tf.keras.initializers.Zeros()),
                layers.Dense(embed_dim, kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=0., stddev=0.02), bias_initializer=tf.keras.initializers.Zeros()),
            ]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output, weights = self.att(
            inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)
        return output, weights


## -------- RESNET BLOCKS ---------- ###

def identity_block(X_input, f1, f2, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # First component of main path
    X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1),
                      padding='valid', name=conv_name_base + '1th',
                      kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = layers.BatchNormalization(axis=-1, name=bn_name_base + '1th')(X)
    X = layers.Activation('relu')(X)

    # Second component of main path
    X = layers.Conv2D(filters=F2, kernel_size=(f1, f2), strides=(1, 1),
                      padding='same', name=conv_name_base + '2nd',
                      kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=-1, name=bn_name_base + '2nd')(X)
    X = layers.Activation('relu')(X)

    # Third component of main path
    X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1),
                      padding='valid', name=conv_name_base + '3rd',
                      kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=-1, name=bn_name_base + '3rd')(X)
    X = layers.Add()([X_input, X])
    X = layers.Activation('relu')(X)
    return X


def convolutional_block(X_input, f1, f2, filters, stage, block, s=2):

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    ##### MAIN PATH #####
    # First component of main path
    X = layers.Conv2D(F1, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1st',
                      kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = layers.BatchNormalization(axis=-1, name=bn_name_base + '1st')(X)
    X = layers.Activation('relu')(X)

    # Second component of main path
    X = layers.Conv2D(F2, (f1, f2), strides=(1, 1), padding='same', name=conv_name_base + '2nd',
                      kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=-1, name=bn_name_base + '2nd')(X)
    X = layers.Activation('relu')(X)

    # Third component of main path
    X = layers.Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '3rd',
                      kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=-1, name=bn_name_base + '3rd')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = layers.Conv2D(F3, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                               kernel_initializer=glorot_uniform(seed=0))(X_input)
    X_shortcut = layers.BatchNormalization(
        axis=-1, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X_shortcut, X])
    X = layers.Activation('relu')(X)

    return X

# if __name__=="__main__":
#     l = TransformerBlock(embed_dim=128, num_heads=3, ff_dim=768)

#     q = tf.random.uniform((2,98,128))

#     l(q,True)
