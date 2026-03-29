"""Module providing onnx2gguf functionality."""

import urllib.request
import urllib.error
import json
from typing import Any, Tuple


def fetch_hf_config(repo_id: str, token: str = None) -> Tuple[dict[str, Any], str, str]:
    """Fetch hf config."""
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
        return None
    try:
        req = urllib.request.Request(f"{url}/tokenizer.json", headers=headers)
        with urllib.request.urlopen(req) as res:
            tokenizer = res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        print(f"Failed to fetch tokenizer: {e}")
    except Exception as e:
        return None
    return (config, tokenizer, f"https://huggingface.co/{repo_id}")


def generate_readme(model_name: str, original_repo: str, quantization: str) -> str:
    """Generate readme."""
    return f'---\nbase_model: {original_repo}\ntags:\n- gguf\n- onnx9000\n---\n\n# {model_name} GGUF\n\nThis model was exported to GGUF format from {original_repo} using the [onnx9000](https://github.com/samuel/onnx9000) zero-dependency compiler.\n\n## Quantization\n- **Level:** {quantization}\n- **Format:** GGUF v3\n\n## Usage\n```bash\nllama-cli -m {model_name.lower()}-{quantization.lower()}.gguf -p "Hello, world!"\n```\n'
