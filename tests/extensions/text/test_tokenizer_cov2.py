import pytest
from onnx9000.extensions.text.tokenizer import Tokenizer


def test_tokenizer_from_huggingface_not_implemented():
    with pytest.raises(NotImplementedError):
        Tokenizer.from_huggingface("path")
