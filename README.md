# Guitar Tone Capture with Autoencoder

This project aims to replicate the functionality of the Mooer Tone Capture GTR pedal using machine learning techniques. Specifically, an Autoencoder will be used to capture the tone (timbre) of a target guitar and apply it to another guitar. The Autoencoder will learn to represent and transfer the timbre between different guitar audio inputs.

## Project Overview

### Goal
To develop a model that captures the timbre of a target guitar and applies it to a source guitar, preserving the original notes and structure of the source guitar while transforming its sound to match the timbre of the target guitar.

### Input
- Audio from the source guitar.

### Output
- Audio from the source guitar with the timbre of the target guitar applied.

---

## Development Stages

### 1. Problem Definition
- **Objective**: Create a machine learning model capable of transferring the timbre of one guitar (target) to another guitar (source).
- **Input**: Audio files of the source guitar.
- **Output**: Modified audio files that sound like the source guitar but with the timbre of the target guitar.

---

### 2. Data Collection
- **Guitar Source Audio**: Collect recordings of the guitar whose sound you want to modify.
- **Guitar Target Audio**: Collect recordings of the guitar whose timbre you want to capture.
  
#### Data Sources
- Recordings from your own equipment.
- Public datasets of guitar sounds (e.g., IDMT-SMT-Guitar Dataset).
- Simulations, if real recordings are unavailable.

---

### 3. Audio Preprocessing
- **Time-Frequency Domain Conversion**: Convert the raw audio into spectrograms or Mel-spectrograms using a Short-Time Fourier Transform (STFT).
- **Normalization**: Normalize the audio data to ensure the amplitudes are within a similar range.
- **Segmentation**: Divide the audio into smaller time windows for processing.

---

### 4. Model Definition and Training

#### Model Architecture: Autoencoder
- **Encoder**: Maps the input audio spectrogram into a lower-dimensional latent space.
- **Latent Space**: This space will represent the timbre characteristics of the target guitar.
- **Decoder**: Reconstructs the audio from the latent space into a spectrogram that reflects the desired timbre.

#### Training Steps:
1. Train the Autoencoder to reconstruct the target guitar's sound.
2. Use a loss function like Mean Squared Error (MSE) to minimize the difference between input and reconstructed spectrograms.
3. Train on diverse audio samples from the target guitar to ensure the model generalizes well.

---

### 5. Timbre Extraction and Transfer
- **Capture the Target Timbre**: Use the trained Encoder to extract the latent representation (timbre) of the target guitar.
- **Apply to Source Guitar**: Feed the source guitar audio through the Encoder and modify the latent space representation before passing it through the trained Decoder to apply the target timbre to the source audio.

---

### 6. Post-Processing of Audio
- **Inverse Transform**: Use the Inverse Short-Time Fourier Transform (ISTFT) to convert the generated spectrogram back into an audio waveform.
- **Filtering and Smoothing**: Optionally apply noise reduction or equalization filters to refine the output audio.

---

### 7. Evaluation and Validation
- **Qualitative Evaluation**: Listen to the generated audio to determine how well the timbre has been transferred.
- **Quantitative Metrics**: Optionally, use similarity metrics between the generated and target spectrograms to evaluate accuracy.

---

### 8. Testing and Refinement
- **Test with Different Inputs**: Try the model with various guitar sounds, such as chords and riffs.
- **Model Adjustments**: Refine the Autoencoder architecture or latent space size if the timbre transfer is not accurate.

---

### 9. Future Extensions
- **Variational Autoencoders (VAEs)**: Test VAEs to increase flexibility in the timbre representation and generation.
- **Additional Controls**: Add user-controlled parameters for fine-tuning timbre transfer.
- **User Interface**: Develop an interface for loading and processing guitar sounds intuitively.

---

## Technologies and Tools
- **Programming Language**: Python.
- **Audio Libraries**: 
  - `librosa` for spectrogram generation and audio processing.
  - `soundfile` or `wave` for audio file I/O.
- **Machine Learning Framework**: 
  - TensorFlow or PyTorch for building the Autoencoder model.
- **Hardware**: GPU for training large models and processing high-resolution audio.

---

## Project Roadmap

1. **Data Collection**: Gather audio samples of source and target guitars.
2. **Preprocessing**: Prepare audio files for training (segmentation, normalization).
3. **Model Training**: Train the Autoencoder on the target guitar sounds.
4. **Testing**: Apply the trained model to transform the source guitar sounds.
5. **Evaluation**: Fine-tune based on qualitative and quantitative results.
6. **Deployment**: Package the solution for practical use, possibly with a GUI for easy interaction.

---

## Acknowledgements
- Thanks to [IDMT-SMT-Guitar Dataset](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/guitar.html) for providing guitar audio samples.
