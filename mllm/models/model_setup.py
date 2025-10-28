"""
Setup functions for language model and tokenizer initialization.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def setup_tokenizer(model_path: str, cache_dir: str = None) -> AutoTokenizer:
    """
    Initialize and configure tokenizer with special tokens.
    Args:
        model_path: Path or name of pretrained model
        cache_dir: Directory to cache model files
    Returns:
        Configured tokenizer with special tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.unk_token # quick hack for Mistral tokenizer
    if tokenizer.pad_token is None:
        pad_token = tokenizer.eos_token
        tokenizer.pad_token = pad_token
        print("tokenizer.pad_token: ", tokenizer.pad_token)
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        print(f"Added pad token: {tokenizer.pad_token}")
    if not hasattr(tokenizer, "start_of_audio"):
        tokenizer.start_of_audio = "AUDIO_START" # create a new special token for Mistral 7B
        tokenizer.add_tokens([tokenizer.start_of_audio])
        print(tokenizer.encode(tokenizer.start_of_audio))
        print("tokenizer.start_of_audio: ", tokenizer.start_of_audio)
    if not hasattr(tokenizer, "eos_token"):
        tokenizer.eos_token = "<|im_end|>"
        print("tokenizer.eos_token: ", tokenizer.eos_token)
    if tokenizer.eos_token == tokenizer.pad_token:
        raise ValueError("tokenizer.eos_token and tokenizer.pad_token are the same, It make infinite generation")
    return tokenizer

def setup_llm(model_path: str, attn_implementation: str, cache_dir: str, dtype: torch.dtype) -> tuple:
    """
    Load language model and tokenizer.
    Args:
        model_path: Path or name of pretrained model
        attn_implementation: Attention implementation type
        cache_dir: Directory to cache model files
        dtype: Data type for model weights
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = setup_tokenizer(model_path, cache_dir)
    lm = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    # hack for Mistral to expand the token embeddings
    lm.resize_token_embeddings(len(tokenizer))
    return lm, tokenizer
