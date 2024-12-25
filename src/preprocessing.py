import librosa
import numpy as np

def audio_to_spectrogram(audio_path, n_fft=1024, hop_length=512):
    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(spectrogram)
    return magnitude, sr

def spectrogram_to_audio(magnitude, sr, n_fft=1024, hop_length=512):
    phase = np.angle(librosa.stft(y))
    spectrogram = magnitude * np.exp(1j * phase)
    audio = librosa.istft(spectrogram, hop_length=hop_length)
    return audio
