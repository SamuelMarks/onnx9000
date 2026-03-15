"""Module providing core logic and structural definitions."""

import json
import struct


def export_tokenizer_binary(json_path: str, output_path: str) -> None:
    """
    Exports a HuggingFace tokenizer.json file into a compressed binary format
    for faster web loading.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", {})
    vocab = model.get("vocab", {})
    merges = model.get("merges", [])
    added = data.get("added_tokens", [])

    for t in added:
        content = t.get("content")
        id_val = t.get("id")
        if content is not None and id_val is not None:
            vocab[content] = int(id_val)

    # For BPE tokenizer, merges list provides the order
    with open(output_path, "wb") as f:
        # Magic bytes
        f.write(b"ONNX9000TK")

        # Vocab size
        f.write(struct.pack("<I", len(vocab)))

        # Vocab items
        for token, token_id in vocab.items():
            token_bytes = token.encode("utf-8")
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)
            f.write(struct.pack("<I", int(token_id)))

        # Merges
        f.write(struct.pack("<I", len(merges)))
        for merge_str in merges:
            parts = merge_str.split(" ")
            if len(parts) == 2:
                p1 = parts[0].encode("utf-8")
                p2 = parts[1].encode("utf-8")
                f.write(struct.pack("<H", len(p1)))
                f.write(p1)
                f.write(struct.pack("<H", len(p2)))
                f.write(p2)
