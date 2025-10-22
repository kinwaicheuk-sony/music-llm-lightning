"""
Audio encoder modules for extracting audio features.
"""
# from .clap import load_clap_audio
from .mel_spec import load_mel_spec

def load_audio_encoder(audio_encoder_type: str):
    """
    Load audio encoder by type.

    Args:
        audio_encoder_type: Type of encoder ('mel_spec' or 'clap')

    Returns:
        Initialized audio encoder module
    """
    if audio_encoder_type == "mel_spec":
        return load_mel_spec()
    else:
        raise ValueError(f"Invalid audio encoder type: {audio_encoder_type}")
