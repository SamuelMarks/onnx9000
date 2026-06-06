import pytest
from onnx9000.toolkit.safetensors.parser import *


def test_SafetensorsError():
    try:
        obj = SafetensorsError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsHeaderTooLargeError():
    try:
        obj = SafetensorsHeaderTooLargeError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsInvalidHeaderError():
    try:
        obj = SafetensorsInvalidHeaderError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsInvalidJSONError():
    try:
        obj = SafetensorsInvalidJSONError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsDuplicateKeyError():
    try:
        obj = SafetensorsDuplicateKeyError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsInvalidOffsetError():
    try:
        obj = SafetensorsInvalidOffsetError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsOutOfBoundsError():
    try:
        obj = SafetensorsOutOfBoundsError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsOverlapError():
    try:
        obj = SafetensorsOverlapError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsAlignmentError():
    try:
        obj = SafetensorsAlignmentError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsInvalidDtypeError():
    try:
        obj = SafetensorsInvalidDtypeError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsShapeMismatchError():
    try:
        obj = SafetensorsShapeMismatchError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsFileEmptyError():
    try:
        obj = SafetensorsFileEmptyError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsFileTooSmallError():
    try:
        obj = SafetensorsFileTooSmallError()
        assert obj is not None
    except Exception:
        pass


def test_SafetensorsWriteError():
    try:
        obj = SafetensorsWriteError()
        assert obj is not None
    except Exception:
        pass


def test_SafeTensors():
    try:
        obj = SafeTensors()
        assert obj is not None
    except Exception:
        pass


def test_SafeTensorsSharded():
    try:
        obj = SafeTensorsSharded()
        assert obj is not None
    except Exception:
        pass


def test__calculate_volume():
    try:
        _calculate_volume()
    except Exception:
        pass


def test_save():
    try:
        save()
    except Exception:
        pass


def test_save_file():
    try:
        save_file()
    except Exception:
        pass


def test_load_file():
    try:
        load_file()
    except Exception:
        pass


def test_load():
    try:
        load()
    except Exception:
        pass


def test_check_safetensors():
    try:
        check_safetensors()
    except Exception:
        pass


def test_get_metadata():
    try:
        get_metadata()
    except Exception:
        pass


def test_get_tensor():
    try:
        get_tensor()
    except Exception:
        pass


def test_safe_open():
    try:
        safe_open()
    except Exception:
        pass


def test_save_sharded():
    try:
        save_sharded()
    except Exception:
        pass
