import json
import base64
from typing import Any


def extract_tokenizer_metadata(
    tokenizer_json_str: str = None, vocab_size: int = 0
) -> dict[str, Any]:
    meta = {}

    if not tokenizer_json_str:
        # 89. Generate a dummy tokenizer if no tokenizer metadata is provided
        meta["tokenizer.ggml.model"] = "llama"
        tokens = [f"[TOKEN_{i}]" for i in range(vocab_size)] if vocab_size > 0 else ["<s>", "</s>"]
        if vocab_size == 0:
            vocab_size = 2
        meta["tokenizer.ggml.tokens"] = tokens
        meta["tokenizer.ggml.scores"] = [0.0] * vocab_size
        meta["tokenizer.ggml.token_type"] = [1] * vocab_size
        meta["tokenizer.ggml.bos_token_id"] = 0
        meta["tokenizer.ggml.eos_token_id"] = 1
        meta["tokenizer.ggml.unknown_token_id"] = 0
        meta["tokenizer.ggml.padding_token_id"] = 0
        meta["tokenizer.ggml.separator_token_id"] = 0
        meta["tokenizer.ggml.add_bos_token"] = True
        meta["tokenizer.ggml.add_eos_token"] = False
        meta["tokenizer.chat_template"] = ""
        return meta

    try:
        t = json.loads(tokenizer_json_str)
    except json.JSONDecodeError:
        # Maybe it's a base64 sentencepiece model
        meta["tokenizer.ggml.model"] = "llama"
        return meta

    model = t.get("model", {})
    model_type = model.get("type", "BPE")

    if model_type == "BPE":
        meta["tokenizer.ggml.model"] = "gpt2"
    elif model_type == "Unigram":
        meta["tokenizer.ggml.model"] = "llama"
    else:
        meta["tokenizer.ggml.model"] = "llama"

    vocab = model.get("vocab", {})
    if isinstance(vocab, dict):
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        tokens = [k for k, v in sorted_vocab]

        # Validate length
        if vocab_size > 0 and len(tokens) != vocab_size:
            # 90. Validate length matches vocab_size exactly
            # If not, pad it or truncate it
            if len(tokens) < vocab_size:
                tokens.extend([f"[DUMMY_{i}]" for i in range(len(tokens), vocab_size)])
            else:
                tokens = tokens[:vocab_size]

        meta["tokenizer.ggml.tokens"] = tokens
        meta["tokenizer.ggml.scores"] = [0.0] * len(tokens)
        meta["tokenizer.ggml.token_type"] = [1] * len(tokens)

    merges = model.get("merges", [])
    if merges:
        meta["tokenizer.ggml.merges"] = merges

    meta["tokenizer.ggml.bos_token_id"] = (
        t.get("added_tokens", [{}])[0].get("id", 0) if t.get("added_tokens") else 0
    )
    meta["tokenizer.ggml.eos_token_id"] = (
        t.get("added_tokens", [{}])[-1].get("id", 1) if t.get("added_tokens") else 1
    )
    meta["tokenizer.ggml.unknown_token_id"] = 0
    meta["tokenizer.ggml.padding_token_id"] = 0
    meta["tokenizer.ggml.separator_token_id"] = 0
    meta["tokenizer.ggml.add_bos_token"] = True
    meta["tokenizer.ggml.add_eos_token"] = False

    # Optional chat template
    meta["tokenizer.chat_template"] = t.get("chat_template", "")

    return meta
