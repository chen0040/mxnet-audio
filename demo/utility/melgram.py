import os
import matplotlib
import sys
import pylab
from matplotlib import pyplot as plt
import librosa
import librosa.display
import numpy as np


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), '..', path)


def melgram_v1(audio_file_path, to_file):
    sig, fs = librosa.load(audio_file_path)

    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(to_file, bbox_inches=None, pad_inches=0)
    pylab.close()


def melgram_v2(audio_file_path):
    # Load sound file
    y, sr = librosa.load(audio_file_path)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.core.amplitude_to_db(S, ref=np.max)

    # Make a new figure
    plt.figure(figsize=(12, 4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()
    plt.show()


def main():
    sys.path.append(patch_path('..'))

    audio_file_path = patch_path('data/audio_samples/example.mp3')
    from mxnet_audio.library.utility.audio_utils import compute_melgram

    arr = compute_melgram(audio_file_path)
    print('melgram: ', arr.shape)


if __name__ == '__main__':
    main()
