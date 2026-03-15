"""Module providing core logic and structural definitions."""

import struct
from typing import Any

from onnx9000.extensions.text.tokenizer import BPETokenizer


class BinaryLoader:
    """Provides semantic functionality and verification."""

    @staticmethod
    def load_bpe_binary(file_path: str, **kwargs: Any) -> BPETokenizer:
        """Provides semantic functionality and verification."""
        with open(file_path, "rb") as f:
            magic = f.read(10)
            if magic != b"ONNX9000TK":
                raise ValueError("Invalid binary format")

            vocab_size_bytes = f.read(4)
            vocab_size = struct.unpack("<I", vocab_size_bytes)[0]

            vocab: dict[str, int] = {}
            for _ in range(vocab_size):
                len_bytes = f.read(2)
                token_len = struct.unpack("<H", len_bytes)[0]
                token_bytes = f.read(token_len)
                token_str = token_bytes.decode("utf-8")
                id_bytes = f.read(4)
                token_id = struct.unpack("<I", id_bytes)[0]
                vocab[token_str] = token_id

            merges_size_bytes = f.read(4)
            merges_size = struct.unpack("<I", merges_size_bytes)[0]

            merges: dict[tuple[str, str], int] = {}
            for idx in range(merges_size):
                len1_bytes = f.read(2)
                len1 = struct.unpack("<H", len1_bytes)[0]
                p1 = f.read(len1).decode("utf-8")

                len2_bytes = f.read(2)
                len2 = struct.unpack("<H", len2_bytes)[0]
                p2 = f.read(len2).decode("utf-8")

                merges[(p1, p2)] = idx

        return BPETokenizer(vocab=vocab, merges=merges, **kwargs)
