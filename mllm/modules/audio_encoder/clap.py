"""
CLAP (Contrastive Language-Audio Pretraining) audio encoder module.
"""
import torch
import os
import ast
import librosa
import laion_clap
import torchaudio

def load_clap_audio() -> 'CUSTOM_CLAP':
    """
    Load CLAP audio encoder.

    Returns:
        Initialized CUSTOM_CLAP model
    """
    model = CUSTOM_CLAP()
    return model

class CUSTOM_CLAP(torch.nn.Module):
    """CLAP-based audio encoder with pretrained weights."""

    def __init__(self) -> None:
        """Initialize CLAP model and load pretrained checkpoint."""
        super().__init__()
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()  # download the default pretrained checkpoint
        self.embed_dim = 512
        self.clap_sample_rate = 48000
        self.remove_text_barnch()
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=48000)

    def remove_text_barnch(self) -> None:
        """Remove text branch components to reduce memory usage."""
        # Remove text_branch, text_transform, text_projection from model.model
        if hasattr(self.model, "model"):
            for attr in ["text_branch", "text_transform", "text_projection"]:
                if hasattr(self.model.model, attr):
                    delattr(self.model.model, attr)

    def get_global_embedding(self, x: torch.Tensor, sample_rate: int = 44100) -> torch.Tensor:
        """
        Extract global audio embedding using CLAP.

        Args:
            x: Audio waveform (B, C, T) or (B, T)
            sample_rate: Input audio sample rate

        Returns:
            Global audio embedding (B, D)
        """
        if x.shape[1] == 2:
            x = x.mean(dim=1, keepdim=False) # change to mono
        if sample_rate == 44100:
            x = self.resampler(x)
        with torch.no_grad():
            embed = self.model.get_audio_embedding_from_data(
                x=x, use_tensor=True
        )
        return embed
