import torch
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np

# Custom Dataset for Spectrograms
class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, sr=16000, n_fft=1024, hop_length=512):
        self.file_paths = file_paths
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Load audio and convert to spectrogram
        y, _ = librosa.load(file_path, sr=self.sr)
        spectrogram = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))
        # Normalize
        spectrogram_db = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
        return torch.tensor(spectrogram_db, dtype=torch.float32)

# List of file paths (replace with your dataset paths)
file_paths = ["audio1.wav", "audio2.wav", "audio3.wav"]  # Example audio files

# Create Dataset and DataLoader
dataset = SpectrogramDataset(file_paths)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)  # Batch size of 16