import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile


def log_spectrogram(
    audio: str,
    sample_rate: int,
    window_size: int = 40,
    step_size: int = 20,
    eps: int = 1e-10,
):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
    )
    return freqs, np.log(spec.T.astype(np.float32) + eps)


def audio2spectrogram(filepath: str):
    samplerate, test_sound = wavfile.read(filepath, mmap=True)
    _, spectrogram = log_spectrogram(test_sound, samplerate)
    return spectrogram


def audio2wave(filepath: str):
    _ = plt.figure(figsize=(5, 5))
    samplerate, test_sound = wavfile.read(filepath, mmap=True)
    plt.plot(test_sound)
    plt.show()
