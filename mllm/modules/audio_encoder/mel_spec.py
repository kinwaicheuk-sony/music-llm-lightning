import os
import torch
import torchaudio

def load_mel_spec():
    model = MELSpec()
    return model

class MELSpec(torch.nn.Module):
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = int(0.01 * 44100), n_mels: int = 128, f_min: int = 0, f_max: int = 22050, power: int = 1):
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

    def get_local_embedding(self, x: torch.Tensor):
        if x.shape[1] == 2:
            x = x.mean(dim=1, keepdim=False) # change to mono
        h_audio = self.mel_spec(x) # B x D x T
        h_audio = h_audio.transpose(1, 2).contiguous() # B x T x D
        return h_audio

    def get_global_embedding(self, x: torch.Tensor):
        if x.shape[1] == 2:
            x = x.mean(dim=1, keepdim=False) # change to mono
        h_audio = self.mel_spec(x) # B x D x T
        h_audio = h_audio.transpose(1, 2).contiguous() # B x T x D
        h_audio = h_audio.mean(dim=1, keepdim=True) # B x D
        return h_audio
