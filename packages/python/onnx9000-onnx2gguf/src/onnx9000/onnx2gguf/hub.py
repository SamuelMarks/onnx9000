import urllib.request
import urllib.error
import json
from typing import Any, Tuple


def fetch_hf_config(repo_id: str, token: str = None) -> Tuple[dict[str, Any], str, str]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://huggingface.co/{repo_id}/resolve/main"

    config = {}
    tokenizer = ""

    try:
        req = urllib.request.Request(f"{url}/config.json", headers=headers)
        with urllib.request.urlopen(req) as res:
            config = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"Failed to fetch config: {e}")
    except Exception as e:
        pass

    try:
        req = urllib.request.Request(f"{url}/tokenizer.json", headers=headers)
        with urllib.request.urlopen(req) as res:
            tokenizer = res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        print(f"Failed to fetch tokenizer: {e}")
    except Exception as e:
        pass

    return config, tokenizer, f"https://huggingface.co/{repo_id}"


def generate_readme(model_name: str, original_repo: str, quantization: str) -> str:
    return f"""---
base_model: {original_repo}
tags:
- gguf
- onnx9000
---

# {model_name} GGUF

This model was exported to GGUF format from {original_repo} using the [onnx9000](https://github.com/samuel/onnx9000) zero-dependency compiler.

## Quantization
- **Level:** {quantization}
- **Format:** GGUF v3

## Usage
```bash
llama-cli -m {model_name.lower()}-{quantization.lower()}.gguf -p "Hello, world!"
```
"""
