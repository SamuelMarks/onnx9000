"""Provide functionality for this module."""


class TokenizerStream:
    """Stream decoder for real-time generation."""

    def __init__(self, tokenizer: "Tokenizer"):
        """Initialize the instance."""
        self.tokenizer = tokenizer
        self.token_cache: list[int] = []

    def put(self, token_id: int) -> str:
        """Add a token and return any completed text."""
        self.token_cache.append(token_id)
        # simplistic for now: just decode all and return diff
        text = self.tokenizer.decode(self.token_cache)
        # in a real BPE we'd handle partial utf-8
        # returning full text for the mock implementation
        return text


class Tokenizer:
    """Base Tokenizer interface."""

    def __init__(self, added_tokens: dict[str, int] = None):
        """Initialize the instance."""
        self.added_tokens = added_tokens or {}
        self.inv_added_tokens = {v: k for k, v in self.added_tokens.items()}

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], clean_up_tokenization_spaces: bool = True) -> str:
        """Decode token IDs to text."""
        text = "".join(chr(t) for t in token_ids)
        if clean_up_tokenization_spaces:
            import re

            text = re.sub(r" +", " ", text).strip()
        return text

    def id_to_token(self, token_id: int) -> str:
        """Provide token ID to string lookup utilities."""
        if hasattr(self, "inv_added_tokens") and token_id in self.inv_added_tokens:
            return self.inv_added_tokens[token_id]
        return "".join(chr(token_id))

    def token_to_id(self, token: str) -> int:
        """Execute the token_to_id operation."""
        return ord(token[0]) if token else 0

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Batched encoding."""
        return [self.encode(t) for t in texts]

    def decode_batch(self, batch_ids: list[list[int]]) -> list[str]:
        """Batched decoding."""
        return [self.decode(ids) for ids in batch_ids]

    def create_stream(self) -> TokenizerStream:
        """Create a decoding stream for real-time output."""
        return TokenizerStream(self)


class BPETokenizer(Tokenizer):
    """Byte-Pair Encoding (BPE) algorithm implementation."""

    def __init__(
        self,
        merges: list[tuple[str, str]],
        vocab: dict[str, int],
        unk_token: str = "<unk>",
    ):
        """Initialize the instance."""
        self.merges = merges
        self.vocab = vocab
        self.unk_token = unk_token
        self.unk_token_id = self.vocab.get(unk_token, 0)
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def _get_pairs(self, word: tuple[str, ...]) -> set[tuple[str, str]]:
        """Execute the _get_pairs operation."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def encode(self, text: str) -> list[int]:
        # Minimal BPE pre-tokenization
        """Execute the encode operation."""
        words = text.split()
        token_ids = []
        for word in words:
            # Add </w> to end of word typically, but simplistic implementation here
            w = tuple(word)
            if len(w) == 0:
                continue

            while True:
                pairs = self._get_pairs(w)
                if not pairs:
                    break

                # Find best merge pair
                best_pair = None
                for merge in self.merges:
                    if merge in pairs:
                        best_pair = merge
                        break

                if not best_pair:
                    break

                # Apply merge
                new_word = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == best_pair[0] and w[i + 1] == best_pair[1]:
                        new_word.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_word.append(w[i])
                        i += 1
                w = tuple(new_word)

            for token in w:
                token_ids.append(self.vocab.get(token, self.unk_token_id))

        return token_ids

    def decode(self, token_ids: list[int], clean_up_tokenization_spaces: bool = True) -> str:
        """Execute the decode operation."""
        text = "".join([self.inv_vocab.get(tid, self.unk_token) for tid in token_ids])
        if clean_up_tokenization_spaces:
            import re

            text = re.sub(r" +", " ", text).strip()
        return text


class WordPieceTokenizer(Tokenizer):
    """WordPiece tokenization algorithm implementation."""

    def __init__(
        self,
        vocab: dict[str, int],
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
    ):
        """Initialize the instance."""
        self.vocab = vocab
        self.unk_token = unk_token
        self.unk_token_id = self.vocab.get(unk_token, 0)
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_input_chars_per_word = max_input_chars_per_word

    def encode(self, text: str) -> list[int]:
        """Execute the encode operation."""
        words = text.split()
        token_ids = []
        for word in words:
            if len(word) > self.max_input_chars_per_word:
                token_ids.append(self.unk_token_id)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(word):
                end = len(word)
                cur_substr = None
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(self.vocab[cur_substr])
                start = end

            if is_bad:
                token_ids.append(self.unk_token_id)
            else:
                token_ids.extend(sub_tokens)

        return token_ids

    def decode(self, token_ids: list[int], clean_up_tokenization_spaces: bool = True) -> str:
        """Execute the decode operation."""
        text = ""
        for tid in token_ids:
            token = self.inv_vocab.get(tid, self.unk_token)
            if token.startswith("##"):
                text += token[2:]
            else:
                if text:
                    text += " "
                text += token

        if clean_up_tokenization_spaces:
            import re

            text = re.sub(r" +", " ", text).strip()

        return text


class UnigramTokenizer(Tokenizer):
    """Unigram tokenization algorithm implementation."""

    def __init__(self, vocab: dict[str, float], unk_token: str = "<unk>"):
        """Initialize the instance."""
        self.vocab = vocab
        self.unk_token = unk_token
        self.unk_score = self.vocab.get(unk_token, -100.0)
        self.token_to_id_map = {k: i for i, k in enumerate(vocab.keys())}
        self.id_to_token_map = {i: k for k, i in self.token_to_id_map.items()}
        self.unk_token_id = self.token_to_id_map.get(unk_token, 0)

    def encode(self, text: str) -> list[int]:
        # Simplistic Viterbi/Unigram scoring simulation
        """Execute the encode operation."""
        words = text.split()
        token_ids = []
        for word in words:
            if not word:
                continue
            n = len(word)
            best_scores = [float("-inf")] * (n + 1)
            best_scores[0] = 0.0
            backpointers = [0] * (n + 1)

            for i in range(1, n + 1):
                for j in range(0, i):
                    sub = word[j:i]
                    if sub in self.vocab:
                        score = best_scores[j] + self.vocab[sub]
                        if score > best_scores[i]:
                            best_scores[i] = score
                            backpointers[i] = j

            if best_scores[n] == float("-inf"):
                token_ids.append(self.unk_token_id)
            else:
                curr = n
                subs = []
                while curr > 0:
                    prev = backpointers[curr]
                    subs.append(word[prev:curr])
                    curr = prev
                subs.reverse()
                for sub in subs:
                    token_ids.append(self.token_to_id_map[sub])

        return token_ids

    def decode(self, token_ids: list[int], clean_up_tokenization_spaces: bool = True) -> str:
        """Execute the decode operation."""
        text = " ".join([self.id_to_token_map.get(tid, self.unk_token) for tid in token_ids])
        if clean_up_tokenization_spaces:
            import re

            text = re.sub(r" +", " ", text).strip()
        return text


class HuggingFaceTokenizerLoader:
    """Loads HuggingFace tokenizer.json formats."""

    @staticmethod
    def load_from_json(json_content: str) -> Tokenizer:
        """Execute the load_from_json operation."""
        import json

        data = json.loads(json_content)
        model = data.get("model", {})
        model_type = model.get("type", "")

        if model_type == "BPE":
            vocab = model.get("vocab", {})
            # Mock parsing of merges which are typically 'A B'
            merges_raw = model.get("merges", [])
            merges = [tuple(m.split(" ", 1)) for m in merges_raw]
            unk_token = model.get("unk_token", "<unk>")
            return BPETokenizer(merges=merges, vocab=vocab, unk_token=unk_token)

        elif model_type == "WordPiece":
            vocab = model.get("vocab", {})
            unk_token = model.get("unk_token", "[UNK]")
            max_input_chars_per_word = model.get("max_input_chars_per_word", 100)
            return WordPieceTokenizer(
                vocab=vocab,
                unk_token=unk_token,
                max_input_chars_per_word=max_input_chars_per_word,
            )

        elif model_type == "Unigram":
            vocab_list = model.get("vocab", [])
            vocab = {item[0]: item[1] for item in vocab_list}
            unk_token = model.get("unk_token", "<unk>")
            return UnigramTokenizer(vocab=vocab, unk_token=unk_token)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class UnicodeNormalizer:
    """Apply Unicode normalization (NFC, NFD, NFKC, NFKD)."""

    @staticmethod
    def normalize(text: str, form: str = "NFC") -> str:
        """Execute the normalize operation."""
        import unicodedata

        if form not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError(f"Unsupported normalization form: {form}")
        return unicodedata.normalize(form, text)


class PreTokenizer:
    """Handles pre-tokenization strategies (whitespace, punctuation, byte-level)."""

    @staticmethod
    def whitespace_split(text: str) -> list[str]:
        """Execute the whitespace_split operation."""
        import re

        return re.findall(r"\S+|\s+", text)

    @staticmethod
    def punctuation_split(text: str) -> list[str]:
        """Execute the punctuation_split operation."""
        import re

        # Splits on punctuation but keeps it
        return re.findall(r"[\w\s]+|[^\w\s]", text)

    @staticmethod
    def byte_level(text: str) -> list[str]:
        # Translates string into byte tokens mapped to specific character sets (e.g. GPT-2 style)
        # This is a mock translation, in real systems it uses a constant 256 mapping
        """Execute the byte_level operation."""
        return [chr(b) for b in text.encode("utf-8")]
