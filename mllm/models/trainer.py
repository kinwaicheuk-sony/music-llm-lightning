"""
Music LLM training wrapper using PyTorch Lightning.
Implements multimodal training with audio encoder and adapter modules.
"""
import os
import random
import json
import datetime
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from mllm.common_utils.config_utils import instantiate_from_config
from mllm.models.model_setup import setup_llm
from mllm.modules.audio_encoder import load_audio_encoder
from mllm.modules.adapter import load_adapter

def load_mllm(ckpt_path: str, ckpt_config_path: str, inference_only: bool = True) -> 'LMTrainingWrapper':
    """
    Load trained music LLM model from checkpoint.
    Args:
        ckpt_path: Path to model checkpoint file
        ckpt_config_path: Path to configuration YAML file
        inference_only: Whether to load for inference only (frees memory)
    Returns:
        Loaded LMTrainingWrapper model
    """
    exp_config = OmegaConf.load(ckpt_config_path)
    model_params = exp_config["model"]["params"]
    model = LMTrainingWrapper.load_from_checkpoint(
        ckpt_path,
        model_config=model_params["model_config"],
        optimizer_config=model_params["optimizer_config"],
        scheduler_config=model_params["scheduler_config"],
        map_location="cpu"
    )
    if inference_only:
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    return model

class LMTrainingWrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper for training multimodal music LLM.
    Combines audio encoder, adapter, and language model.
    """

    def __init__(self, model_config=None, optimizer_config=None, scheduler_config=None):
        """
        Initialize training wrapper with model, optimizer, and scheduler configs.

        Args:
            model_config: Configuration for model architecture
            optimizer_config: Configuration for optimizer
            scheduler_config: Configuration for learning rate scheduler
        """
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.model_type = model_config.model_type
        self.expdir = model_config.expdir
        self.max_length = model_config.max_length
        self.adapter_type = model_config.adapter_type
        self.target_dtype = self.setup_precision()
        self.lm, self.tokenizer = setup_llm(
            self.model_config.model_path,
            self.model_config.attn_implementation,
            self.model_config.cache_dir,
            self.target_dtype
        )
        self.start_of_audio = self.tokenizer.start_of_audio
        self.audio_token_id = self.tokenizer.encode(self.start_of_audio)
        self.lm_hidden_size = self.lm.config.hidden_size
        self.audio_encoder = load_audio_encoder(self.model_config.audio_encoder_type)
        self.adapter = load_adapter(self.adapter_type, self.model_config.adapter_config)

        self.lm.to(self.target_dtype)
        self.audio_encoder.to(self.target_dtype)
        self.adapter.to(self.target_dtype)
        self.lm.eval()
        self.audio_encoder.eval()
        self.adapter.train()
        for param in self.lm.parameters():
            param.requires_grad = False
        if self.model_config.apply_lora:
            self.apply_lora()
        self.validation_step_outputs = []
        self.inference_data = {}

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for language modeling.
        Args:
            logits: Model output logits (B, T, V)
            labels: Target token IDs (B, T)
        Returns:
            Cross-entropy loss value
        """
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100
        )

    def apply_lora(self) -> None:
        """Apply LoRA (Low-Rank Adaptation) to the language model."""
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=self.model_config.lora_rank,  # Rank of LoRA update matrices
            lora_alpha=(self.model_config.lora_rank * 2),  # Alpha scaling factor
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.lm = get_peft_model(self.lm, lora_config)
        self.lm.print_trainable_parameters()

    def setup_precision(self) -> torch.dtype:
        """
        Setup model precision based on config.
        Returns:
            torch.dtype for model weights
        """
        if self.model_config.precision in ["bf16", "bf16-mixed"]:
            target_dtype = torch.bfloat16
        elif self.model_config.precision in ["32", "32-mixed"]:
            target_dtype = torch.float32
        return target_dtype

    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def dtype(self):
        return list(self.parameters())[0].dtype

    def get_audio_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract audio embeddings using encoder and adapter.
        Args:
            audio: Raw audio tensor (B, C, T)
        Returns:
            Adapted audio embeddings (B, T, D)
        """
        with torch.no_grad():
            h_audio = self.audio_encoder.get_local_embedding(audio)
            h_audio = h_audio.detach() # for stop gradient
        z_audio = self.adapter(h_audio)
        return z_audio

    def training_multimodal_embedding(self, input_ids: torch.Tensor, output_ids: torch.Tensor,
                                     input_attention_mask: torch.Tensor, output_attention_mask: torch.Tensor,
                                     z_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combine text and audio embeddings for training.
        Args:
            input_ids: Input token IDs
            output_ids: Output token IDs
            input_attention_mask: Input attention mask
            output_attention_mask: Output attention mask
            z_audio: Audio embeddings

        Returns:
            Tuple of (batch_embeds, batch_attention_mask, batch_labels)
        """
        audio_token_id = torch.tensor([self.audio_token_id], device=self.device)
        audio_token_id = audio_token_id[:,2:] # quick hack for Mistral tokenizer producing 3 tokens instead of 1
        audio_insert_index = torch.where(input_ids == audio_token_id)[1].tolist()
        if self.model_config.apply_lora:
            inputs_embeds = self.lm.model.model.embed_tokens(input_ids).to(self.device)
            output_embeds = self.lm.model.model.embed_tokens(output_ids).to(self.device)
        else:
            inputs_embeds = self.lm.model.embed_tokens(input_ids).to(self.device)
            output_embeds = self.lm.model.embed_tokens(output_ids).to(self.device)
        audio_attention_mask = torch.ones(z_audio.shape[:2], device=self.device)
        batch_embeds = []
        batch_attention_mask = []
        batch_labels = []

        for batch_idx, audio_insert_idx in enumerate(audio_insert_index):
            embeds = torch.cat([inputs_embeds[batch_idx, :audio_insert_idx], z_audio[batch_idx, :], inputs_embeds[batch_idx, audio_insert_idx+1:], output_embeds[batch_idx, :]], dim=0)
            attention_mask = torch.cat([input_attention_mask[batch_idx, :audio_insert_idx], audio_attention_mask[batch_idx, :], input_attention_mask[batch_idx, audio_insert_idx+1:], output_attention_mask[batch_idx, :]], dim=0)
            labels = torch.full(embeds.shape[:1], fill_value=-100, device=self.device, dtype=torch.long)
            labels[-output_embeds.shape[1]:] = output_ids[batch_idx, :] # alpaca-style loss function
            batch_embeds.append(embeds)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        batch_embeds = torch.stack(batch_embeds)
        batch_attention_mask = torch.stack(batch_attention_mask)
        batch_labels = torch.stack(batch_labels)
        pad_mask = (batch_labels == self.tokenizer.pad_token_id)  # Find padding tokens
        batch_labels[pad_mask] = -100  # Replace pad tokens with -100
        return batch_embeds[:,:self.max_length], batch_attention_mask[:,:self.max_length], batch_labels[:,:self.max_length] # for training stability

    def inference_multimodal_embedding(self, input_ids: torch.Tensor, input_attention_mask: torch.Tensor,
                                      z_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine text and audio embeddings for inference.
        Args:
            input_ids: Input token IDs
            input_attention_mask: Input attention mask
            z_audio: Audio embeddings

        Returns:
            Tuple of (batch_embeds, batch_attention_mask)
        """
        audio_token_id = torch.tensor([self.audio_token_id], device=self.device)
        audio_token_id = audio_token_id[:,2:] # quick hack for Mistral tokenizer producing 3 tokens instead of 1
        audio_insert_index = torch.where(input_ids == audio_token_id)[1].tolist()
        if self.model_config.apply_lora:
            inputs_embeds = self.lm.model.model.embed_tokens(input_ids).to(self.device)
        else:
            inputs_embeds = self.lm.model.embed_tokens(input_ids).to(self.device)
        audio_attention_mask = torch.ones(z_audio.shape[:2], device=self.device)
        batch_embeds = []
        batch_attention_mask = []
        for batch_idx, audio_insert_idx in enumerate(audio_insert_index):
            embeds = torch.cat([inputs_embeds[batch_idx, :audio_insert_idx], z_audio[batch_idx, :], inputs_embeds[batch_idx, audio_insert_idx+1:]], dim=0)
            attention_mask = torch.cat([input_attention_mask[batch_idx, :audio_insert_idx], audio_attention_mask[batch_idx, :], input_attention_mask[batch_idx, audio_insert_idx+1:]], dim=0)
            batch_embeds.append(embeds)
            batch_attention_mask.append(attention_mask)
        batch_embeds = torch.stack(batch_embeds)
        batch_attention_mask = torch.stack(batch_attention_mask)
        return batch_embeds, batch_attention_mask

    def forward_pass(self, batch: dict) -> torch.Tensor:
        """
        Forward pass through the model for one batch.
        Args:
            batch: Dictionary containing 'input_text', 'output_text', and 'audio'
        Returns:
            Loss value
        """
        input_tokens = self.tokenizer(
            batch['input_text'],
            return_tensors="pt",
            max_length=self.model_config.input_length,
            padding="max_length",
            truncation=True,
            padding_side="left"
        )
        output_tokens = self.tokenizer(
            batch['output_text'],
            return_tensors="pt",
            max_length=self.model_config.output_length,
            padding="max_length",
            truncation=True,
            padding_side="right"
        )
        input_ids = input_tokens.input_ids.to(self.device)
        input_attention_mask = input_tokens.attention_mask.to(self.device)
        output_ids = output_tokens.input_ids.to(self.device)
        output_attention_mask = output_tokens.attention_mask.to(self.device)
        z_audio = self.get_audio_embedding(batch['audio'].to(self.dtype))
        inputs_embeds, attention_mask, labels = self.training_multimodal_embedding(
            input_ids, output_ids, input_attention_mask, output_attention_mask, z_audio
        )
        outputs = self.lm.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1, :].contiguous() # shift for next token prediction
        labels = labels[:, 1:].contiguous() # shift for next token prediction
        loss = self.loss_fn(logits, labels)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step for one batch.

        Args:
            batch: Training batch data
            batch_idx: Batch index

        Returns:
            Loss value
        """
        loss = self.forward_pass(batch)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def generation_monitoring(self, batch: dict, global_step: int) -> list[str]:
        """
        Generate text from audio for monitoring during training.

        Args:
            batch: Batch with audio and input text
            global_step: Current training step

        Returns:
            List of generated text outputs
        """
        input_tokens = self.tokenizer(
            batch['input_text'], # instruction
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        input_ids = input_tokens.input_ids
        input_attention_mask = input_tokens.attention_mask
        z_audio = self.get_audio_embedding(batch['audio'].to(self.dtype))
        inputs_embeds, attention_mask = self.inference_multimodal_embedding(
            input_ids, input_attention_mask, z_audio
        )
        outputs = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=128, # because song describer case
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        os.makedirs(f"{self.expdir}/monitor", exist_ok=True)
        with open(f"{self.expdir}/monitor/{global_step}.json", "w") as f:
            json.dump({"input": batch['input_text'], "output": output_texts[0]}, f, indent=2)
        return output_texts

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Validation step for one batch.

        Args:
            batch: Validation batch data
            batch_idx: Batch index

        Returns:
            Dictionary with validation loss
        """
        loss = self.forward_pass(batch)
        self.log("val_loss", loss, sync_dist=True)
        # Store the loss in the outputs dictionary
        self.validation_step_outputs = getattr(self, "validation_step_outputs", [])
        self.validation_step_outputs.append({"val_loss": loss})
        # Store the inference data
        self.inference_data = getattr(self, "inference_data", [])
        random_idx = random.randint(0, len(batch['input_text']) - 1)
        self.inference_data = {"input_text": batch['input_text'][random_idx:random_idx+1], "audio": batch['audio'][random_idx:random_idx+1]}
        return {"val_loss": loss}

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch to log metrics and save weights."""
        global_step = self.global_step
        output_texts = self.generation_monitoring(self.inference_data, global_step)
        # Calculate average validation loss using stored outputs
        val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        # Calculate perplexity
        # Log metrics
        self.log("mean_val_loss", val_loss, sync_dist=True)
        # Clear the outputs to free memory
        self.validation_step_outputs = []
        self.inference_data = {}
        if self.model_config.save_adapter:
            weight_dict = self.adapter.state_dict()
            save_weight_dict = weight_dict.copy()
            for key in save_weight_dict:
                save_weight_dict[key] = save_weight_dict[key].detach()
            torch.save(save_weight_dict, os.path.join(self.expdir, f"{self.adapter_type}_weights.pth"))

    def configure_optimizers(self) -> tuple[list, list]:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Tuple of ([optimizer], [scheduler_config])
        """
        optimizer = instantiate_from_config(
            self.optimizer_config, params=filter(lambda p: p.requires_grad, self.parameters())
        )
        scheduler = instantiate_from_config(
            self.scheduler_config, optimizer=optimizer
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]


