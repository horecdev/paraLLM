from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = Path("./models/qwen2.5-0.5b-instruct")
FILES = ["config.json", "tokenizer.json", "model.safetensors"]

def main():
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  for filename in FILES:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=OUT_DIR,
    )
    print(f"Downloaded {filename} -> {local_path}")


if __name__ == "__main__":
  main()