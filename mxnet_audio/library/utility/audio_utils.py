import librosa
import numpy as np
from math import floor
import os


def compute_melgram(audio_path):
    """ Compute a mel-spectrogram and returns it in a shape of (96,1366, 1), where
       96 == #mel-bins and 1366 == #time frame
    """

    mg_path = audio_path + '-mg.npy'

    if os.path.exists(mg_path):
        return np.load(mg_path)

    print('computing mel-spetrogram for audio: ', audio_path)

    # mel-spectrogram parameters
    sampling_rate = 12000
    n_fft = 512
    n_mels = 96
    hop_length = 256
    duration_in_seconds = 29.12  # to make it 1366 frame (1366 = 12000 * 29.12 / 256)

    src, sr = librosa.load(audio_path, sr=sampling_rate)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(duration_in_seconds * sampling_rate)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(duration_in_seconds * sampling_rate) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
    logam = librosa.core.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=sampling_rate, hop_length=hop_length,
                        n_fft=n_fft, n_mels=n_mels) ** 2,
                ref=1.0)
    ret = np.expand_dims(ret, axis=2)

    np.save(mg_path, ret)

    return ret


def compute_melgram_multiframe(audio_path, all_song=True):
    ''' Compute a mel-spectrogram in multiple frames of the song and returns it in a shape of (N,1,96,1366), where
    96 == #mel-bins, 1366 == #time frame, and N=#frames
    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..
    if all_song:
        DURA_TRASH = 0
    else:
        DURA_TRASH = 20

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)
    n_sample_trash = int(DURA_TRASH * SR)

    # remove the trash at the beginning and at the end
    src = src[n_sample_trash:(n_sample - n_sample_trash)]
    n_sample = n_sample - 2 * n_sample_trash

    ret = np.zeros((0, 1, 96, 1366), dtype=np.float32)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
        logam = librosa.core.amplitude_to_db
        melgram = librosa.feature.melspectrogram
        ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                            n_fft=N_FFT, n_mels=N_MELS) ** 2,
                    ref=1.0)
        ret = ret[np.newaxis, np.newaxis, :]

    elif n_sample > n_sample_fit:  # if too long
        N = int(floor(n_sample // n_sample_fit))

        src_total = src

        for i in range(0, N):
            src = src_total[(i * n_sample_fit):(i + 1) * (n_sample_fit)]

            logam = librosa.core.amplitude_to_db
            melgram = librosa.feature.melspectrogram
            retI = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                                 n_fft=N_FFT, n_mels=N_MELS) ** 2,
                         ref=1.0)
            retI = retI[np.newaxis, np.newaxis, :]
            ret = np.concatenate((ret, retI), axis=0)

    return ret
