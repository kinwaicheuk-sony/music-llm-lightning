import os
import json
import torch
from torch.utils.data import Dataset
from mllm.models.model_setup import setup_tokenizer
from mllm.common_utils.audio_utils.audio_io import load_audio

class AudioTextDataset(Dataset):
    def __init__(self, data_dir, model_path, target_duration=10, split="train", cache_dir=None):
        self.data_dir = data_dir
        self.target_duration = target_duration
        self.split = split
        self.tokenizer = setup_tokenizer(model_path)
        self.start_of_audio = self.tokenizer.start_of_audio # we use Qwen Vision Start token for audio, for bypass vocabulary expansion
        self.instruction = "Describe the music in detail."
        self.eos_token = self.tokenizer.eos_token
        self.metadata = self._load_metadata()

    def __len__(self):
        return len(self.metadata)

    def _load_metadata(self):
        metadata = []
        for file in os.listdir(f"{self.data_dir}/json"):
            if file.endswith(".json"):
                data = json.load(open(os.path.join(self.data_dir, "json", file) , 'r'))
                metadata.append(data)
        if self.split == "train":
            metadata = metadata[:1000]
        else:
            metadata = metadata[1000:]
        return metadata

    def _apply_chat_template(self, instruction):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{instruction}{self.start_of_audio}"}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def __getitem__(self, idx):
        # example: {'id': 1093, 'text': 'A rock song...', 'audio_path': '02/880702.mp3'}
        row = self.metadata[idx]
        audio, sr = load_audio(os.path.join(self.data_dir, "audio", row["audio_path"]))
        input_text = self._apply_chat_template(instruction = self.instruction)
        output_text = row['text'] + self.eos_token
        return {
            "input_text": input_text,
            "output_text": output_text,
            "audio": torch.from_numpy(audio),
        }
