# Music LLM Lightning: Finetuning LLMs for Music Understanding

A PyTorch Lightning-based framework for finetuning Large Language Models (LLMs) on music captioning tasks. This project enables audio-to-text generation by connecting audio encoders to pretrained language models through trainable adapter modules.

## Overview

This framework implements a multimodal language model that processes audio inputs and generates descriptive text captions. The architecture consists of three main components:

- **Audio Encoder**: Extracts acoustic features from music (Mel-Spectrogram or CLAP)
- **Adapter Module**: Projects audio features into the LLM's embedding space
- **Language Model**: Pretrained LLM (Qwen3-0.6B) for text generation

### Key Features

- Multiple audio encoder options (Mel-Spectrogram, CLAP)
- Flexible adapter architectures (Linear, MLP, Q-Former)
- Parameter-efficient finetuning with LoRA
- Distributed training with PyTorch Lightning
- Configurable via YAML files
- Built-in learning rate scheduling (Linear Warmup + Cosine Decay)
- Automatic checkpoint saving and TensorBoard logging

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-compatible GPU (recommended)
- `uv` package manager (or `pip`)

### Setup

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv add torch torchvision torchaudio lightning
uv pip install -e .
```

### Dependencies

The project requires the following key packages:
- `torch>=2.8.0`, `torchaudio>=2.8.0`
- `lightning>=2.5.5`
- `transformers>=4.57.1`
- `peft>=0.17.1` (for LoRA)
- `librosa>=0.11.0`, `soundfile>=0.13.1`
- `omegaconf>=2.3.0`, `einops>=0.8.1`

## Dataset Preparation

The project uses the Song Describer dataset from HuggingFace.

```bash
# Download and extract dataset
bash scripts/fetch_data.sh
python mllm/preprocessing/song_describer.py
```

This will download audio files and JSON metadata to the `./data` directory.

### Dataset Structure

```
data/
├── audio/
│   ├── 00/
│   │   ├── 1051200.mp3
│   │   └── ...
│   └── ...
└── json/
    ├── 1.json
    └── ...
```

Each JSON file contains:
```json
{
  "id": 1,
  "text": "Weird synth sounds that are kind of relaxing...",
  "audio_path": "93/945193.mp3"
}
```

## Training

### Basic Training

```bash
python train.py --config config/default.yaml
```

### Training Options

```bash
# Specify GPU
python train.py --config config/default.yaml --gpu 0

# Resume from checkpoint
python train.py --config config/default.yaml --resume path/to/checkpoint.ckpt
```

### Monitoring

During training inferenc example save in `./exp/llm/default/monitor`

```
{
  "input": [
    "<|im_start|>user\nDescribe the music in detail.<|vision_start|><|im_end|>\n<|im_start|>assistant\n"
  ],
  "output": "A smooth jazz rendition of a Christmas song that makes me feel calm and happy."
}
```

### Inference

After training your model, you can use `infer.py` to generate text descriptions from audio files.

#### Basic Usage

Activate the virtual environment and run inference:

```bash
source .venv/bin/activate
python infer.py \
  --ckpt_path ./exp/llm/default/last.ckpt \
  --ckpt_config_path ./exp/llm/default/config.yaml \
  --audio_path ./data/audio/93/945193.mp3 \
  --input_text "Describe the song in detail." \
  --device cuda
```

```bash
This song is electronic, with a synth and drum beat and a melodic synth. It starts with instruments then male vocaals are added.
```


### Configuration

Training is configured via YAML files. Key parameters in `config/default.yaml`:

#### Model Configuration

```yaml
model_type: qwen3                  # LLM architecture
model_path: Qwen/Qwen3-0.6B       # HuggingFace model path
audio_encoder_type: mel_spec       # mel_spec or clap
adapter_type: mlp                  # linear, mlp, or q_former
```

#### Adapter Configuration

```yaml
adapter_config:
  i_dim: 128      # Input dimension (mel_spec n_mels)
  o_dim: 1024     # Output dimension (LLM hidden size)
```

#### LoRA Configuration

```yaml
apply_lora: True
lora_rank: 128
```

LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`

#### Training Hyperparameters

```yaml
steps: 100000
batch_size: 4
lr: 5e-5
warmup_steps: 1000
max_length: 2048      # Total sequence length
input_length: 16      # Instruction length
output_length: 512    # Caption length
```

#### Hardware Configuration

```yaml
precision: "32"           # or "bf16-mixed" for mixed precision
activation_checkpointing: True
accelerator: gpu
strategy: ddp            # or fsdp
```

## Architecture Details

### Model Pipeline

1. **Audio Processing**: 10-second audio clips are loaded at 44.1kHz
2. **Feature Extraction**: Mel-Spectrogram (128 mel bins) computed with 10ms hop length
3. **Adapter Projection**: Audio features projected to LLM embedding space
4. **Multimodal Fusion**: Audio embeddings inserted at `<|vision_start|>` token position
5. **Text Generation**: Autoregressive generation conditioned on audio

### Training Strategy

- **Frozen Components**: Audio encoder and base LLM parameters
- **Trainable Components**: Adapter module and LoRA weights
- **Loss Function**: Cross-entropy on caption tokens (Alpaca-style)
- **Optimization**: AdamW with fused kernels
- **Scheduler**: Linear warmup (1k steps) + Cosine decay

### Instruction Format

The model uses Qwen's chat template:

```python
instruction = "Describe the music in detail."
input_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": f"{instruction}<|vision_start|>"}],
    add_generation_prompt=True,
    tokenize=False
)
```

## Project Structure

```
music-llm-lightning/
├── mllm/
│   ├── models/
│   │   ├── trainer.py          # LightningModule wrapper
│   │   ├── model_setup.py      # LLM initialization
│   │   └── data/
│   │       ├── pt_dataset.py   # PyTorch dataset
│   │       └── pl_modules.py   # Lightning data module
│   ├── modules/
│   │   ├── audio_encoder/
│   │   │   ├── mel_spec.py     # Mel-Spectrogram encoder
│   │   │   └── clap.py         # CLAP encoder
│   │   └── adapter/
│   │       ├── linear.py       # Linear projection
│   │       ├── mlp.py          # MLP adapter
│   │       └── q_former.py     # Q-Former adapter
│   └── common_utils/
│       ├── audio_utils/        # Audio I/O utilities
│       ├── scheduler.py        # Learning rate schedulers
│       └── config_utils.py     # Config instantiation
├── config/
│   └── default.yaml            # Default configuration
├── scripts/
│   └── fetch_data.sh          # Data download script
├── train.py                    # Training script
└── pyproject.toml             # Package dependencies
```

## Outputs

### Checkpoints

Checkpoints are saved to `./exp/llm/{model_name}/`:
- `last.ckpt`: Latest checkpoint
- `{epoch}-{step}-{val_loss:.2f}.ckpt`: Best validation checkpoints
- `mlp_weights.pth`: Adapter weights only

### Logs

TensorBoard logs are saved to `./exp/llm/{model_name}/`:

```bash
tensorboard --logdir ./exp/llm/
```

Logged metrics:
- `train_loss`: Training loss per step
- `val_loss`: Validation loss per epoch
- `learning_rate`: Current learning rate

## Inference

To load a trained model for inference:

```python
from mllm.models.trainer import load_mllm

model = load_mllm(
    ckpt_path="exp/llm/default/last.ckpt",
    ckpt_config_path="exp/llm/default/config.yaml",
    inference_only=True
)
```

## Customization

### Adding a New Audio Encoder

1. Create encoder class in `mllm/modules/audio_encoder/`:

```python
class MyEncoder(torch.nn.Module):
    def get_local_embedding(self, x: torch.Tensor):
        # Return [B, T, D] features
        pass
```

2. Register in `mllm/modules/audio_encoder/__init__.py`

### Adding a New Adapter

1. Create adapter class in `mllm/modules/adapter/`:

```python
class MyAdapter(nn.Module):
    def __init__(self, i_dim, o_dim):
        super().__init__()
        # Define layers

    def forward(self, x):
        return x  # [B, T, o_dim]
```

2. Register in `mllm/modules/adapter/__init__.py`

## Performance Tips

### Memory Optimization
- Enable gradient checkpointing: `activation_checkpointing: True`
- Reduce batch size or sequence length
- Use mixed precision: `precision: "bf16-mixed"`
- Use FSDP for large models: `strategy: fsdp`

### Speed Optimization
- Enable TF32: `tf32: True` (automatic in default config)
- Use fused AdamW: `fused: True`
- Increase `num_workers` for data loading
- Enable persistent workers: `persistent_workers: True`
