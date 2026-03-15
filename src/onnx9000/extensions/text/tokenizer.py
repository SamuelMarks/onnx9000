"""Module providing core logic and structural definitions."""

import json
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Optional, Union


class Tokenizer(ABC):
    """
    Foundational Tokenizer base class for pure Python text tokenization.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
    ) -> None:
        """Provides semantic functionality and verification."""
        self.vocab: dict[str, int] = vocab
        self.inverse_vocab: dict[int, str] = {v: k for k, v in vocab.items()}
        self.unk_token: str = unk_token
        self.pad_token: str = pad_token
        self.cls_token: str = cls_token
        self.sep_token: str = sep_token
        self.mask_token: str = mask_token

        self.unk_id: int = self.vocab.get(self.unk_token, -1)
        self.pad_id: int = self.vocab.get(self.pad_token, -1)
        self.cls_id: int = self.vocab.get(self.cls_token, -1)
        self.sep_id: int = self.vocab.get(self.sep_token, -1)
        self.mask_id: int = self.vocab.get(self.mask_token, -1)

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encodes a string into a list of token IDs."""
        return []

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs into a string."""
        return ""

    def encode_to_buffer(self, text: str) -> memoryview:
        """Encodes directly to an Int32Array (memoryview in Python)."""
        import ctypes

        ids = self.encode(text)
        arr = (ctypes.c_int32 * len(ids))(*ids)
        return memoryview(arr)

    def encode_plus(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
        truncation: bool = False,
    ) -> dict[str, list[int]]:
        """
        Encode with padding, truncation, attention masks, and token type IDs.
        """
        ids1: list[int] = self.encode(text)
        ids2: list[int] = self.encode(text_pair) if text_pair else []

        # Sequence pair formatting: [CLS] text [SEP] text_pair [SEP]
        input_ids: list[int] = []
        token_type_ids: list[int] = []

        if self.cls_id != -1:
            input_ids.append(self.cls_id)
            token_type_ids.append(0)

        input_ids.extend(ids1)
        token_type_ids.extend([0] * len(ids1))

        if self.sep_id != -1:
            input_ids.append(self.sep_id)
            token_type_ids.append(0)

        if ids2:
            input_ids.extend(ids2)
            token_type_ids.extend([1] * len(ids2))
            if self.sep_id != -1:
                input_ids.append(self.sep_id)
                token_type_ids.append(1)

        if truncation and max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            token_type_ids = token_type_ids[:max_length]
            if self.sep_id != -1 and input_ids[-1] != self.sep_id:
                input_ids[-1] = self.sep_id

        attention_mask: list[int] = [1] * len(input_ids)

        if pad_to_max_length and max_length is not None and len(input_ids) < max_length:
            pad_len: int = max_length - len(input_ids)
            input_ids.extend([self.pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            token_type_ids.extend([0] * pad_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def normalize(
        self, text: str, lower: bool = True, strip_accents: bool = True
    ) -> str:
        """
        Normalize text natively.
        """
        if lower:
            text = text.lower()
        if strip_accents:
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return text

    def pre_tokenize(self, text: str) -> list[str]:
        """
        Pre-tokenization regex splitting.
        """
        # Basic whitespace and punctuation split
        tokens: list[str] = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    @classmethod
    def from_huggingface(
        cls, path_or_json: Union[str, dict[str, object]]
    ) -> "Tokenizer":
        """
        Parse huggingface tokenizer.json configuration.
        """
        raise NotImplementedError("Subclasses must implement from_huggingface.")


class BPETokenizer(Tokenizer):
    """
    Byte-Pair Encoding (BPE) greedy merging algorithm in pure Python.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: dict[tuple[str, str], int],
        byte_fallback: bool = False,
        dropout_prob: float = 0.0,
        **kwargs: object,
    ) -> None:
        """Provides semantic functionality and verification."""
        super().__init__(vocab, **kwargs)  # type: ignore
        self.merges: dict[tuple[str, str], int] = merges
        self.cache: dict[str, list[str]] = {}
        self.byte_fallback: bool = byte_fallback
        self.dropout_prob: float = dropout_prob

    def _get_pairs(self, word: list[str]) -> set[tuple[str, str]]:
        """Return set of symbol pairs in a word."""
        pairs: set[tuple[str, str]] = set()
        if len(word) < 2:
            return pairs
        prev_char: str = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token: str) -> list[str]:
        """Apply BPE to a single token."""
        # Disable cache if using dropout to maintain randomness
        if self.dropout_prob == 0.0 and token in self.cache:
            return self.cache[token]

        word: list[str] = list(token)
        if not word:
            return []

        pairs: set[tuple[str, str]] = self._get_pairs(word)
        if not pairs:
            return [token]

        import random

        while True:
            bigram: Optional[tuple[str, str]] = None
            min_rank: float = float("inf")
            for pair in pairs:
                rank: int = self.merges.get(pair, int(1e9))
                if rank < min_rank:
                    min_rank = rank
                    bigram = pair

            if bigram is None or bigram not in self.merges:
                break

            if self.dropout_prob > 0.0 and random.random() < self.dropout_prob:
                # remove this bigram from possible merges to try the next best one
                pairs.remove(bigram)
                continue

            first, second = bigram
            new_word: list[str] = []
            i: int = 0
            while i < len(word):
                try:
                    j: int = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        if self.dropout_prob == 0.0:
            self.cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        """Encode text to BPE token IDs."""
        text = self.normalize(text, lower=False, strip_accents=False)
        tokens: list[str] = self.pre_tokenize(text)
        ids: list[int] = []
        for token in tokens:
            bpe_tokens: list[str] = self.bpe(token)
            for bpe_token in bpe_tokens:
                if bpe_token in self.vocab:
                    ids.append(self.vocab[bpe_token])
                else:
                    if self.byte_fallback:
                        for byte in bpe_token.encode("utf-8"):
                            byte_str = f"<0x{byte:02X}>"
                            ids.append(self.vocab.get(byte_str, self.unk_id))
                    else:
                        ids.append(self.unk_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode BPE token IDs to text."""
        text_tokens: list[str] = []
        for i in ids:
            if i in self.inverse_vocab:
                text_tokens.append(self.inverse_vocab[i])
            else:
                text_tokens.append(self.unk_token)

        # Simple detokenization reconstruct for BPE
        text: str = "".join(text_tokens)
        text = text.replace("Ġ", " ")
        text = text.replace(" ", " ")
        return text.strip()

    @classmethod
    def from_huggingface(
        cls, path_or_json: Union[str, dict[str, object]]
    ) -> "BPETokenizer":
        """Provides semantic functionality and verification."""
        if isinstance(path_or_json, str):
            with open(path_or_json, encoding="utf-8") as f:
                data: dict[str, object] = json.load(f)
        else:
            data = dict(path_or_json)

        model_data: dict[str, object] = data.get("model", {})  # type: ignore
        vocab: dict[str, int] = model_data.get("vocab", {})  # type: ignore

        merges_list: list[str] = model_data.get("merges", [])  # type: ignore
        merges: dict[tuple[str, str], int] = {}
        for idx, merge_str in enumerate(merges_list):
            parts = merge_str.split(" ")
            if len(parts) == 2:
                merges[(parts[0], parts[1])] = idx

        # Handle Added Tokens
        added_tokens: list[dict[str, object]] = data.get("added_tokens", [])  # type: ignore
        for token_data in added_tokens:
            content: str = str(token_data.get("content", ""))
            id_val: int = int(str(token_data.get("id", -1)))
            if content and id_val != -1:
                vocab[content] = id_val

        return cls(vocab=vocab, merges=merges)


class WordPieceTokenizer(Tokenizer):
    """
    WordPiece tokenization logic in pure Python.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        max_input_chars_per_word: int = 100,
        **kwargs: object,
    ) -> None:
        """Provides semantic functionality and verification."""
        super().__init__(vocab, **kwargs)  # type: ignore
        self.max_input_chars_per_word: int = max_input_chars_per_word

    def encode(self, text: str) -> list[int]:
        """Encode text using WordPiece."""
        text = self.normalize(text)
        tokens: list[str] = self.pre_tokenize(text)
        output_ids: list[int] = []

        for token in tokens:
            chars: list[str] = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_ids.append(self.unk_id)
                continue

            is_bad: bool = False
            start: int = 0
            sub_tokens: list[int] = []
            while start < len(chars):
                end: int = len(chars)
                cur_substr: Optional[str] = None
                cur_id: int = -1
                while start < end:
                    substr: str = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        cur_id = self.vocab[substr]
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_id)
                start = end

            if is_bad:
                output_ids.append(self.unk_id)
            else:
                output_ids.extend(sub_tokens)

        return output_ids

    def decode(self, ids: list[int]) -> str:
        """Decode WordPiece token IDs to text."""
        tokens: list[str] = []
        for i in ids:
            tokens.append(self.inverse_vocab.get(i, self.unk_token))

        out_str: str = ""
        for token in tokens:
            if token.startswith("##"):
                out_str += token[2:]
            else:
                if out_str:
                    out_str += " "
                out_str += token
        return out_str
