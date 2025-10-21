import os
import torch
import random
import numpy as np
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .pt_dataset import AudioTextDataset

class AudioTextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        model_path: str,
        batch_size: int = 256,
        num_workers: int = 8,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir

    def prepare_data(self) -> None:
        """Validate that required files exist."""

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = AudioTextDataset(
                data_dir=self.data_dir,
                model_path=self.model_path,
                cache_dir=self.cache_dir,
                split="train",
            )
            # Load validation data
            self.val_dataset = AudioTextDataset(
                data_dir=self.data_dir,
                model_path=self.model_path,
                cache_dir=self.cache_dir,
                split="test",
            )
        elif stage == "test":
            self.test_dataset = AudioTextDataset(
                data_dir=self.data_dir,
                model_path=self.model_path,
                cache_dir=self.cache_dir,
                split="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None
        )

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """Initialize worker with different random seed."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
