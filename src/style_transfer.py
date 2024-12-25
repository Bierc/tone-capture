
from models import *


def style_transfer(input_spectrogram, source_style, target_style):
    input_spectrogram = input_spectrogram.to(device)
    source_style = torch.tensor(source_style).to(device)
    target_style = torch.tensor(target_style).to(device)

    mu, logvar = encoder(input_spectrogram, source_style)
    z = reparameterize(mu, logvar)
    z += encoder.style_embedding(target_style) - encoder.style_embedding(source_style)
    transferred_spectrogram = decoder(z)
    return transferred_spectrogram
