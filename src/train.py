import torch
import torch.nn as nn
from models import *
from utils import *


# Hyperparameters
latent_dim = 128
input_dim = 1024  # Size of spectrogram (flattened)
hidden_dim = 512
num_styles = 5  # Number of instrument styles
lr = 1e-4
epochs = 50

# Check if GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
encoder = Encoder(input_dim, latent_dim, hidden_dim, num_styles)
decoder = Decoder(latent_dim, input_dim, hidden_dim)
discriminator = Discriminator(input_dim, hidden_dim)

# Optimizers
opt_enc = torch.optim.Adam(encoder.parameters(), lr=lr)
opt_dec = torch.optim.Adam(decoder.parameters(), lr=lr)
opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr)

for epoch in range(epochs):
    for x, style in train_loader:  # x: real spectrograms, style: instrument labels
        x = x.to(device)
        style = style.to(device)

        # VAE Pathway
        mu, logvar = encoder(x, style)
        z = reparameterize(mu, logvar)
        x_recon = decoder(z)

        # Discriminator Pathway
        d_real = discriminator(x)  # Real spectrograms
        d_fake = discriminator(x_recon.detach())  # Fake spectrograms (detach to avoid updating generator)

        # Losses
        vae_loss_val = vae_loss(x_recon, x, mu, logvar)
        disc_loss = gan_loss(d_real, d_fake)
        gen_loss = nn.BCELoss()(d_fake, torch.ones_like(d_fake))  # Fool discriminator

        # Backpropagation
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        (vae_loss_val + gen_loss).backward()  # VAE + Generator losses
        opt_enc.step()
        opt_dec.step()

        opt_disc.zero_grad()
        disc_loss.backward()  # Discriminator loss
        opt_disc.step()

    print(f"Epoch {epoch+1}/{epochs}, VAE Loss: {vae_loss_val.item():.4f}, Disc Loss: {disc_loss.item():.4f}")

