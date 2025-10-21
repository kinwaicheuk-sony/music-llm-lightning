import os
import numpy as np
import librosa
import soundfile as sf

def load_audio(input_path, channel_first=True, random_sample=True, duration=10, target_sr=44100):
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

def save_audio(data, samplerate, output_path, audio_format="flac"):
    if data.shape[0] == 2 or data.shape[0] == 1:
        data = data.T # (2, N) -> (N, 2)
    sf.write(output_path, data, samplerate, format=audio_format)
