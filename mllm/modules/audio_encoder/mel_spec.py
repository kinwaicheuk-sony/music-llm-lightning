"""
Mel-spectrogram audio encoder module.
"""
import os
import torch
import torchaudio

def load_mel_spec() -> 'MELSpec':
    """
    Load mel-spectrogram encoder.

    Returns:
        Initialized MELSpec model
    """
    model = MELSpec()
    return model

class MELSpec(torch.nn.Module):
    """Mel-spectrogram audio feature extractor."""

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = int(1 * 44100),
                 n_mels: int = 128, f_min: int = 0, f_max: int = 22050, power: int = 1):
        """
        Initialize mel-spectrogram transform.

        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size
            hop_length: Hop length between frames
            n_mels: Number of mel filterbanks
            f_min: Minimum frequency
            f_max: Maximum frequency
            power: Exponent for magnitude spectrogram
        """
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )

    def get_local_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract local (time-varying) mel-spectrogram features.

        Args:
            x: Audio waveform (B, C, T) or (B, T)

        Returns:
            Mel-spectrogram features (B, T, D)
        """
        if x.shape[1] == 2:
            x = x.mean(dim=1, keepdim=False) # change to mono
        h_audio = self.mel_spec(x) # B x D x T
        h_audio = h_audio.transpose(1, 2).contiguous() # B x T x D
        return h_audio

    def get_global_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global (time-averaged) mel-spectrogram features.

        Args:
            x: Audio waveform (B, C, T) or (B, T)

        Returns:
            Averaged mel-spectrogram features (B, 1, D)
        """
        if x.shape[1] == 2:
            x = x.mean(dim=1, keepdim=False) # change to mono
        h_audio = self.mel_spec(x) # B x D x T
        h_audio = h_audio.transpose(1, 2).contiguous() # B x T x D
        h_audio = h_audio.mean(dim=1, keepdim=True) # B x D
        return h_audio
