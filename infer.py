"""
Inference script for music LLM.
Loads trained model and generates text descriptions from audio.
"""
import os
import torch
import numpy as np
from mllm.common_utils.audio_utils.audio_io import load_audio
from mllm.models.trainer import load_mllm
import argparse

def load_mllm_lightning(ckpt_path: str, ckpt_config_path: str, inference_only: bool = True,device: str = "cuda"):
    """
    Load music LLM from checkpoint for inference.

    Args:
        ckpt_path: Path to checkpoint file
        ckpt_config_path: Path to config YAML file
        inference_only: Whether to optimize for inference
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    mllm_lightning = load_mllm(ckpt_path, ckpt_config_path, inference_only=inference_only)
    mllm_lightning.eval()
    mllm_lightning.to(device)
    return mllm_lightning

def audio_to_text(input_text: str, audio_path: str, mllm_lightning) -> str:
    """
    Generate text description from audio file.

    Args:
        input_text: Instruction/prompt text
        audio_path: Path to audio file
        mllm_lightning: Loaded model

    Returns:
        Generated text description
    """
    tokenizer = mllm_lightning.tokenizer
    lm = mllm_lightning.lm
    device = mllm_lightning.device
    audio, sr = load_audio(audio_path)
    x_audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
    input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{input_text}{mllm_lightning.start_of_audio}"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_tokens = tokenizer(
        input_text, # instruction
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    input_ids = input_tokens.input_ids
    input_attention_mask = input_tokens.attention_mask
    z_audio = mllm_lightning.get_audio_embedding(x_audio)
    inputs_embeds, attention_mask = mllm_lightning.inference_multimodal_embedding(
        input_ids, input_attention_mask, z_audio
    )
    outputs = lm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=128,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_texts[0]

def inference(args: argparse.Namespace) -> None:
    """
    Run inference on audio file.

    Args:
        args: Command-line arguments containing paths and settings
    """
    device = args.device
    mllm_lightning = load_mllm_lightning(args.ckpt_path, args.ckpt_config_path, inference_only=True, device=device)
    output_text = audio_to_text(args.input_text, args.audio_path, mllm_lightning)
    print(output_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="./exp/llm/default/last.ckpt")
    parser.add_argument("--ckpt_config_path", type=str, default="./exp/llm/default/config.yaml")
    parser.add_argument("--audio_path", type=str, default="./data/audio/93/945193.mp3")
    parser.add_argument("--input_text", type=str, default="Describe the song in detail.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    inference(args)
