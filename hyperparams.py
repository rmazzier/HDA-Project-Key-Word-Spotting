import pathlib

# Preprocessing Hyperparameters
SAMPLE_RATE = 16000
FFT_SIZE = 512
WIN_SIZE = 400 # 25 ms
HOP_SIZE = 160 # 10 ms
N_FILTERS = 30 # number of mel filterbanks
N_CEPSTRAL = 13
DELTAS = True # whether to stack the delta MFCCS (and the accelerations as well) together with the actual mfccs.

# Extra Class names
_UNKNOWN_CLASS_ = 'filler'
_SILENCE_CLASS_ = 'silence'

# Directories
_NOISE_DIR_ = pathlib.Path('data/speech_commands_v0.02/_background_noise_')
_DATA_DIR_ = pathlib.Path('data/speech_commands_v0.02')
_BINARIES_DIR_ = pathlib.Path('data/binaries')
_MODELS_DIR_ = pathlib.Path('models')

# ID string for each task
_TASKS_ = ['10kws+U+S', '20kws+U', '35kws']

