"""
Audio I/O utilities for loading and saving audio files.
"""
import os
import numpy as np
import librosa
import soundfile as sf

def load_audio(input_path: str, channel_first: bool = True, random_sample: bool = True,
               duration: int = 10, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """
    Load audio file with optional random cropping and resampling.

    Args:
        input_path: Path to audio file
        channel_first: If True, return shape (C, T), else (T, C)
        random_sample: If True, randomly crop audio
        duration: Duration in seconds to extract
        target_sr: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    y, sr = sf.read(input_path)
    if random_sample:
        start_idx = np.random.randint(0, len(y) - int(duration*sr))
        y = y[start_idx:start_idx+int(duration*sr)]
    if channel_first and y.ndim > 1:
        y = y.T
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type='kaiser_fast')
        sr = target_sr
    return y, sr

def save_audio(data: np.ndarray, samplerate: int, output_path: str, audio_format: str = "flac") -> None:
    """
    Save audio array to file.

    Args:
        data: Audio array (C, T) or (T, C)
        samplerate: Sample rate in Hz
        output_path: Output file path
        audio_format: Audio format (e.g., 'flac', 'wav')
    """
    if data.shape[0] == 2 or data.shape[0] == 1:
        data = data.T # (2, N) -> (N, 2)
    sf.write(output_path, data, samplerate, format=audio_format)
