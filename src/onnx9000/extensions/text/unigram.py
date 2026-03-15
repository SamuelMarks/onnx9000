"""Module providing core logic and structural definitions."""

import json
from typing import Any, Union

from .tokenizer import Tokenizer


class TrieNode:
    """Provides semantic functionality and verification."""

    def __init__(self) -> None:
        """Provides semantic functionality and verification."""
        self.children: dict[str, TrieNode] = {}
        self.is_word: bool = False
        self.score: float = 0.0
        self.id: int = -1


class Trie:
    """Provides semantic functionality and verification."""

    def __init__(self) -> None:
        """Provides semantic functionality and verification."""
        self.root = TrieNode()

    def insert(self, word: str, score: float, token_id: int) -> None:
        """Provides semantic functionality and verification."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.score = score
        node.id = token_id

    def get_prefixes(self, text: str, start: int) -> list[tuple[int, float, int]]:
        """Provides semantic functionality and verification."""
        prefixes: list[tuple[int, float, int]] = []
        node = self.root
        i = start
        while i < len(text):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_word:
                prefixes.append((i + 1, node.score, node.id))
            i += 1
        return prefixes


class UnigramTokenizer(Tokenizer):
    """Provides semantic functionality and verification."""

    def __init__(
        self,
        vocab: dict[str, int],
        scores: dict[str, float],
        unk_score: float = -100.0,
        **kwargs: Any,
    ) -> None:
        """Provides semantic functionality and verification."""
        super().__init__(vocab, **kwargs)
        self.scores: dict[str, float] = scores
        self.unk_score: float = unk_score
        self.trie = Trie()
        for token, idx in self.vocab.items():
            score = self.scores.get(token, self.unk_score)
            self.trie.insert(token, score, idx)

    def encode(self, text: str) -> list[int]:
        """Provides semantic functionality and verification."""
        text = self.normalize(text)
        n = len(text)
        if n == 0:
            return []

        # dp[i] is the best score to tokenize text[:i]
        dp: list[float] = [-float("inf")] * (n + 1)
        # parent[i] is the length of the last token in the best tokenization of text[:i]
        parent: list[tuple[int, int]] = [(0, self.unk_id)] * (n + 1)
        dp[0] = 0.0

        for i in range(n):
            if dp[i] == -float("inf"):
                continue

            prefixes = self.trie.get_prefixes(text, i)
            # Default fallback for single character UNK
            if dp[i] + self.unk_score > dp[i + 1]:
                dp[i + 1] = dp[i] + self.unk_score
                parent[i + 1] = (i, self.unk_id)

            for end, score, token_id in prefixes:
                new_score = dp[i] + score
                if new_score > dp[end]:
                    dp[end] = new_score
                    parent[end] = (i, token_id)

        ids: list[int] = []
        curr = n
        while curr > 0:
            prev, token_id = parent[curr]
            ids.append(token_id)
            curr = prev

        ids.reverse()
        return ids

    def decode(self, ids: list[int]) -> str:
        """Provides semantic functionality and verification."""
        tokens: list[str] = [self.inverse_vocab.get(i, self.unk_token) for i in ids]
        text = "".join(tokens)
        text = text.replace(" ", " ")
        return text.strip()

    @classmethod
    def from_huggingface(
        cls, path_or_json: Union[str, dict[str, Any]]
    ) -> "UnigramTokenizer":
        """Provides semantic functionality and verification."""
        if isinstance(path_or_json, str):
            with open(path_or_json, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = path_or_json

        model_data = data.get("model", {})
        vocab_list = model_data.get("vocab", [])

        vocab: dict[str, int] = {}
        scores: dict[str, float] = {}
        for idx, (token, score) in enumerate(vocab_list):
            vocab[token] = idx
            scores[token] = score

        return cls(vocab=vocab, scores=scores)
