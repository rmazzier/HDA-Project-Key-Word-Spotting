import tensorflow as tf
import math


class MelSpectrogram(tf.keras.layers.Layer):
    """Compute mel_spectrogram from waveform."""

    def __init__(self, sample_rate, fft_size, win_size, hop_size, n_filters,
                 f_min=0.0, f_max=None, **kwargs):
        super(MelSpectrogram, self).__init__(**kwargs)
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
        super(MelSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        mfccs : (tf.Tensor), shape = (None, audio_frames, n_cepstral, ch)
            The corresponding batch of log-mel-spectrograms
        """

        # compute spectrogram with STFT
        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.win_size,
                                      frame_step=self.hop_size,
                                      fft_length=self.fft_size)
        # get absolute value and transpose
        spectrograms = tf.abs(spectrograms)

        # for some reason this passage gives a strange result...
        #spectrogram=tf.multiply(tf.math.square(spectrogram), 1/_FFT_SIZE)

        # map from linear frequency scale to mel scale

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_filters,
            num_spectrogram_bins=self.fft_size//2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)

        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        return tf.expand_dims(log_mel_spectrograms, -1)

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
        config.update(super(MelSpectrogram, self).get_config())

        return config


class MFCC(tf.keras.layers.Layer):
    """Compute mfcc from waveform."""

    def __init__(self, sample_rate, fft_size, win_size, hop_size, n_filters, n_cepstral,
                 f_min=0.0, f_max=None, **kwargs):
        super(MFCC, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.win_size = win_size
        self.hop_size = hop_size
        self.n_filters = n_filters
        self.n_cepstral = n_cepstral
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
        super(MFCC, self).build(input_shape)

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
        def tf_lift(mfccs, L=22):
            """Applies liftering to the mfccs matrix."""
            n = tf.range(mfccs.shape[2], dtype=tf.float32)
            lift = 1.0 + (L/2.0)*tf.math.sin(math.pi*n/L)
            return mfccs * lift

        log_mel_spectrograms = MelSpectrogram(sample_rate=self.sample_rate, fft_size=self.fft_size,
                                              win_size=self.win_size, hop_size=self.hop_size, n_filters=self.n_filters)(waveforms)

        # Now compute MFCCs from log-magnitude mel scaled spectrogram
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[
            :, :self.n_cepstral]
        mfccs = tf_lift(mfccs)
        return tf.expand_dims(mfccs, -1)

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'win_size': self.win_size,
            'n_filters': self.n_filters,
            'n_cepstral': self.n_cepstral,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(MFCC, self).get_config())

        return config
