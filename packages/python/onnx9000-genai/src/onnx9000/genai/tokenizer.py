"""Generative AI Tokenizer implementations and stream decoders.

This module provides high-performance tokenizer architectures (BPE, WordPiece,
Unigram) and real-time streaming decoding utilities compatible with ONNX9000
GenAI pipelines.
"""

from onnx9000.core.exceptions import TokenizationError


class TokenizerStream:
    """Stream decoder for real-time generation."""

    def __init__(self, tokenizer: "Tokenizer") -> None:
        """Initialize the TokenizerStream with a specific tokenizer instance.

        Args:
            tokenizer: The underlying Tokenizer instance used for decoding.
        """
        self.tokenizer = tokenizer
        self.token_cache: list[int] = []

    def put(self, token_id: int) -> str:
        """Add a token ID to the stream and decode any newly completed text.

        Args:
            token_id: The integer ID of the token to append to the cache.

        Returns:
            The decoded text corresponding to the added token.
        """
        self.token_cache.append(token_id)
        # simplistic for now: just decode all and return diff
        text = self.tokenizer.decode(self.token_cache)
        # in a real BPE we'd handle partial utf-8
        # returning full text for the mock implementation
        return text


class Tokenizer:
    """Abstract base class representing the tokenizer interface.

    Provides common encoding, decoding, batching, and stream generation
    APIs for sub-classes (BPE, WordPiece, Unigram).
    """

    def __init__(self, added_tokens: dict[str, int] | None = None) -> None:
        """Initialize the base Tokenizer with optional special/added tokens.

        Args:
            added_tokens: A dictionary mapping string tokens to their integer IDs.
        """
        self.added_tokens = added_tokens or {}
        self.inv_added_tokens = {v: k for k, v in self.added_tokens.items()}

    def encode(self, text: str) -> list[int]:
        """Encode the input text into a list of token IDs.

        Args:
            text: The raw string text to be encoded.

        Returns:
            A list of decoded token integer IDs.
        """
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], clean_up_tokenization_spaces: bool = True) -> str:
        """Decode a list of token IDs back into a single text string.

        Args:
            token_ids: A list of integer token IDs.
            clean_up_tokenization_spaces: If True, cleans up extra/redundant spaces
                in the decoded string. Defaults to True.

        Returns:
            The decoded string.
        """
        text = "".join(chr(t) for t in token_ids)
        if clean_up_tokenization_spaces:
            import re

            text = re.sub(r" +", " ", text).strip()
        return text

    def id_to_token(self, token_id: int) -> str:
        """Convert a token ID back to its string representation.

        Args:
            token_id: The integer ID of the token.

        Returns:
            The string representation of the token.
        """
        if hasattr(self, "inv_added_tokens") and token_id in self.inv_added_tokens:
            return self.inv_added_tokens[token_id]
        return "".join(chr(token_id))

    def token_to_id(self, token: str) -> int:
        """Convert a string token to its corresponding integer ID.

        Args:
            token: The string token to convert.

        Returns:
            The corresponding token ID as an integer.
        """
        return ord(token[0]) if token else 0

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of text strings into lists of token IDs.

        Args:
            texts: A list of string texts to encode.

        Returns:
            A list of token ID lists, one list per input text.
        """
        return [self.encode(t) for t in texts]

    def decode_batch(self, batch_ids: list[list[int]]) -> list[str]:
        """Decode a batch of token ID lists back into a list of strings.

        Args:
            batch_ids: A list of token ID lists.

        Returns:
            A list of decoded strings, one per input token ID list.
        """
        return [self.decode(ids) for ids in batch_ids]

    def create_stream(self) -> TokenizerStream:
        """Create and return a TokenizerStream for real-time decoding.

        Returns:
            A new TokenizerStream instance wrapping this tokenizer.
        """
        return TokenizerStream(self)


class BPETokenizer(Tokenizer):
    """Byte-Pair Encoding (BPE) algorithm implementation."""

    def __init__(
        self,
        merges: list[tuple[str, str]],
        vocab: dict[str, int],
        unk_token: str = "<unk>",
    ) -> None:
        """Initialize the BPETokenizer with merges, vocabulary, and unknown token.

        Args:
            merges: A list of string pairs representing BPE merge operations.
            vocab: A dictionary mapping string tokens to their integer IDs.
            unk_token: The string representing the unknown token. Defaults to "<unk>".
        """
        self.merges = merges
        self.vocab = vocab
        self.unk_token = unk_token
        self.unk_token_id = self.vocab.get(unk_token, 0)
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def _get_pairs(self, word: tuple[str, ...]) -> set[tuple[str, str]]:
        """Extract all adjacent character pairs from a word.

        Args:
            word: A tuple of strings representing characters or sub-tokens.

        Returns:
            A set of string pairs representing adjacent characters.
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def encode(self, text: str) -> list[int]:
        # Minimal BPE pre-tokenization
        """Encode text using the Byte-Pair Encoding (BPE) algorithm.

        Args:
            text: The input text to be tokenized.

        Returns:
            A list of token IDs representing the BPE tokenization of the text.
        """
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
        """Decode a list of BPE token IDs back into a text string.

        Args:
            token_ids: A list of integer token IDs.
            clean_up_tokenization_spaces: If True, cleans up extra/redundant spaces
                in the decoded string. Defaults to True.

        Returns:
            The decoded string.
        """
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
    ) -> None:
        """Initialize the WordPieceTokenizer with vocabulary, unk, and length limits.

        Args:
            vocab: A dictionary mapping string tokens to their integer IDs.
            unk_token: The string representing the unknown token. Defaults to "[UNK]".
            max_input_chars_per_word: The maximum length of a word to attempt
                tokenization on. Defaults to 100.
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.unk_token_id = self.vocab.get(unk_token, 0)
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_input_chars_per_word = max_input_chars_per_word

    def encode(self, text: str) -> list[int]:
        """Encode text using the WordPiece tokenization algorithm.

        Args:
            text: The input text to be tokenized.

        Returns:
            A list of token IDs representing the WordPiece tokenization of the text.
        """
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
        """Decode a list of WordPiece token IDs back into a text string.

        Args:
            token_ids: A list of integer token IDs.
            clean_up_tokenization_spaces: If True, cleans up extra/redundant spaces
                in the decoded string. Defaults to True.

        Returns:
            The decoded string with WordPiece prefix symbols handled correctly.
        """
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

    def __init__(self, vocab: dict[str, float], unk_token: str = "<unk>") -> None:
        """Initialize the UnigramTokenizer with vocab scoring and unknown token.

        Args:
            vocab: A dictionary mapping string tokens to their float probability/scores.
            unk_token: The string representing the unknown token. Defaults to "<unk>".
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.unk_score = self.vocab.get(unk_token, -100.0)
        self.token_to_id_map = {k: i for i, k in enumerate(vocab.keys())}
        self.id_to_token_map = {i: k for k, i in self.token_to_id_map.items()}
        self.unk_token_id = self.token_to_id_map.get(unk_token, 0)

    def encode(self, text: str) -> list[int]:
        # Simplistic Viterbi/Unigram scoring simulation
        """Encode text using the Unigram tokenization algorithm (Viterbi-based).

        Args:
            text: The input text to be tokenized.

        Returns:
            A list of token IDs representing the optimal Unigram segmentation.
        """
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
        """Decode a list of Unigram token IDs back into a text string.

        Args:
            token_ids: A list of integer token IDs.
            clean_up_tokenization_spaces: If True, cleans up extra/redundant spaces
                in the decoded string. Defaults to True.

        Returns:
            The decoded string.
        """
        text = " ".join([self.id_to_token_map.get(tid, self.unk_token) for tid in token_ids])
        if clean_up_tokenization_spaces:
            import re

            text = re.sub(r" +", " ", text).strip()
        return text


class HuggingFaceTokenizerLoader:
    """Loads HuggingFace tokenizer.json formats."""

    @staticmethod
    def load_from_json(json_content: str) -> Tokenizer:
        """Load and instantiate a Tokenizer from a HuggingFace tokenizer.json configuration.

        Args:
            json_content: The JSON string content representing the tokenizer config.

        Returns:
            A Tokenizer instance (BPE, WordPiece, or Unigram) based on the config type.

        Raises:
            ValueError: If the model type in the configuration is unsupported.
        """
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
            raise TokenizationError(f"Unsupported model type: {model_type}")


class UnicodeNormalizer:
    """Apply Unicode normalization (NFC, NFD, NFKC, NFKD)."""

    @staticmethod
    def normalize(text: str, form: str = "NFC") -> str:
        """Normalize the input text according to the specified Unicode normal form.

        Args:
            text: The input string to normalize.
            form: The normal form to apply ("NFC", "NFD", "NFKC", "NFKD").
                Defaults to "NFC".

        Returns:
            The normalized string.

        Raises:
            ValueError: If the normal form is not supported.
        """
        import unicodedata

        if form not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise TokenizationError(f"Unsupported normalization form: {form}")
        return unicodedata.normalize(form, text)


class PreTokenizer:
    """Handles pre-tokenization strategies (whitespace, punctuation, byte-level)."""

    @staticmethod
    def whitespace_split(text: str) -> list[str]:
        """Split the input text into sub-tokens using whitespace boundaries.

        Args:
            text: The raw input string.

        Returns:
            A list of split string tokens including whitespaces.
        """
        import re

        return re.findall(r"\S+|\s+", text)

    @staticmethod
    def punctuation_split(text: str) -> list[str]:
        """Split the input text around punctuation marks while preserving them.

        Args:
            text: The raw input string.

        Returns:
            A list of split string tokens with punctuation preserved as individual items.
        """
        import re

        # Splits on punctuation but keeps it
        return re.findall(r"[\w\s]+|[^\w\s]", text)

    @staticmethod
    def byte_level(text: str) -> list[str]:
        # Translates string into byte tokens mapped to specific character sets (e.g. GPT-2 style)
        # This is a mock translation, in real systems it uses a constant 256 mapping
        """Convert a string into a list of single-byte character representations.

        Args:
            text: The raw input string.

        Returns:
            A list of strings where each string represents a single byte.
        """
        return [chr(b) for b in text.encode("utf-8")]
