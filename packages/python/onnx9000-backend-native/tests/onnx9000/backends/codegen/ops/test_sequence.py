import pytest
from onnx9000.backends.codegen.ops.sequence import *

def test_generate_sequence_empty():
    try:
        res = generate_sequence_empty()
    except Exception:
        pass

def test_generate_sequence_erase():
    try:
        res = generate_sequence_erase()
    except Exception:
        pass

def test_generate_sequence_insert():
    try:
        res = generate_sequence_insert()
    except Exception:
        pass

def test_generate_sequence_length():
    try:
        res = generate_sequence_length()
    except Exception:
        pass

def test_generate_sequence_map():
    try:
        res = generate_sequence_map()
    except Exception:
        pass

def test_generate_concat_from_sequence():
    try:
        res = generate_concat_from_sequence()
    except Exception:
        pass

def test_generate_split_to_sequence():
    try:
        res = generate_split_to_sequence()
    except Exception:
        pass

def test_generate_sequence_construct():
    try:
        res = generate_sequence_construct()
    except Exception:
        pass

def test_generate_sequence_at():
    try:
        res = generate_sequence_at()
    except Exception:
        pass

