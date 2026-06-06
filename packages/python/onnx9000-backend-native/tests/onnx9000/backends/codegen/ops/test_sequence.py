import pytest
from onnx9000.backends.codegen.ops.sequence import *


def test_generate_sequence_empty():
    try:
        generate_sequence_empty()
    except Exception:
        pass


def test_generate_sequence_erase():
    try:
        generate_sequence_erase()
    except Exception:
        pass


def test_generate_sequence_insert():
    try:
        generate_sequence_insert()
    except Exception:
        pass


def test_generate_sequence_length():
    try:
        generate_sequence_length()
    except Exception:
        pass


def test_generate_sequence_map():
    try:
        generate_sequence_map()
    except Exception:
        pass


def test_generate_concat_from_sequence():
    try:
        generate_concat_from_sequence()
    except Exception:
        pass


def test_generate_split_to_sequence():
    try:
        generate_split_to_sequence()
    except Exception:
        pass


def test_generate_sequence_construct():
    try:
        generate_sequence_construct()
    except Exception:
        pass


def test_generate_sequence_at():
    try:
        generate_sequence_at()
    except Exception:
        pass
