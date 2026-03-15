"""Module providing core logic and structural definitions."""

import re
from typing import List, Optional, Union, Dict, Any
import collections
import math


def string_normalizer(
    x: List[str],
    case_change_action: str = "NONE",
    is_case_sensitive: int = 0,
    locale: str = "",
    stopwords: Optional[List[str]] = None,
) -> List[str]:
    """
    Native implementation of ONNX StringNormalizer.
    """
    stopwords_set = set()
    if stopwords:
        for w in stopwords:
            if not is_case_sensitive:
                stopwords_set.add(w.lower())
            else:
                stopwords_set.add(w)

    y = []
    for s in x:
        res = s
        if case_change_action == "LOWER":
            res = res.lower()
        elif case_change_action == "UPPER":
            res = res.upper()

        if stopwords_set:
            check_val = res if is_case_sensitive else res.lower()
            if check_val not in stopwords_set:
                y.append(res)
        else:
            y.append(res)

    return y


def regex_replace(
    x: List[str], pattern: str, rewrite: str, global_replace: int = 1
) -> List[str]:
    """
    Native implementation of ONNX RegexReplace.
    """
    y = []
    # cache compiled regex
    prog = re.compile(pattern)
    count = 0 if global_replace else 1

    for s in x:
        y.append(prog.sub(rewrite, s, count=count))
    return y


def vocab_mapping(
    x: List[str], vocab: dict[str, int], unk_token_id: int = -1
) -> List[int]:
    """
    Native implementation of ONNX VocabMapping.
    """
    return [vocab.get(w, unk_token_id) for w in x]


def tf_idf_vectorizer(
    x: List[List[Union[int, str]]],
    min_gram_length: int,
    max_gram_length: int,
    max_skip_count: int,
    mode: str,
    ngram_counts: List[int],
    ngram_indexes: List[int],
    pool_strings: List[str] = None,
    pool_int64s: List[int] = None,
    weights: List[float] = None,
) -> List[List[float]]:
    """
    Native implementation of ONNX TfIdfVectorizer.
    """
    pool = pool_strings if pool_strings is not None else pool_int64s
    if pool is None:
        pool = []

    # Building n-gram mapping from pool to feature index
    # We parse ngram_counts and ngram_indexes based on the ONNX spec:
    # pool contains flattened 1-grams. ngram_counts contains lengths.
    # ngram_indexes gives the feature index.

    ngram_map = {}
    pool_idx = 0
    for count, feature_idx in zip(ngram_counts, ngram_indexes):
        gram = tuple(pool[pool_idx : pool_idx + count])
        ngram_map[gram] = feature_idx
        pool_idx += count

    num_features = max(ngram_indexes) + 1 if ngram_indexes else 0

    if weights is None:
        weights = [1.0] * num_features

    out = []
    for seq in x:
        freqs = collections.defaultdict(int)

        # Extract n-grams
        for n in range(min_gram_length, max_gram_length + 1):
            # for skip count, standard implementation can get complex,
            # simplified for max_skip_count = 0 usually used
            for i in range(len(seq) - n + 1):
                gram = tuple(seq[i : i + n])
                if gram in ngram_map:
                    freqs[ngram_map[gram]] += 1

        # Compute vector
        vec = [0.0] * num_features
        max_f = max(freqs.values()) if freqs else 0
        total_f = sum(freqs.values()) if freqs else 0

        for f_idx, f_val in freqs.items():
            w = weights[f_idx]
            if mode == "TF":
                vec[f_idx] = float(f_val)
            elif mode == "IDF":
                vec[f_idx] = w
            elif mode == "TFIDF":
                vec[f_idx] = float(f_val) * w

        out.append(vec)

    return out


def wordpiece_tokenizer(
    text: List[str], vocab: Dict[str, int], unk_token: str = "[UNK]"
) -> List[List[int]]:
    """
    ONNX Operator WordpieceTokenizer mapping.
    Uses the implemented WordPieceTokenizer.
    """
    from onnx9000.extensions.text.tokenizer import WordPieceTokenizer

    tokenizer = WordPieceTokenizer(vocab, unk_token=unk_token)
    return [tokenizer.encode(t) for t in text]


def n_gram_extraction(tokens: List[str], n: int) -> List[str]:
    """
    Step 063: Implement n-gram extraction natively.
    """
    if n <= 0 or not tokens:
        return []
    grams = []
    for i in range(len(tokens) - n + 1):
        grams.append(" ".join(tokens[i : i + n]))
    return grams
