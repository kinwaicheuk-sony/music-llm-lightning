import os
import json
from datasets import load_dataset

def main():
    dataset = load_dataset("seungheondoh/eval-song_describer", split="original")
    for item in dataset:
        result = {
            "id": item['caption_id'],
            "text": item['caption'],
            "audio_path": item['path'],
        }
        os.makedirs("data/json", exist_ok=True)
        with open(f"data/json/{item['caption_id']}.json", "w") as f:
            json.dump(result, f)

if __name__ == "__main__":
    main()
