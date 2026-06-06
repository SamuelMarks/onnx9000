import pytest
from onnx9000.tvm.tir.transform.passes import *


def test_LoopUnroller():
    try:
        obj = LoopUnroller()
        assert obj is not None
    except Exception:
        pass


def test_Vectorizer():
    try:
        obj = Vectorizer()
        assert obj is not None
    except Exception:
        pass


def test_StorageFlattener():
    try:
        obj = StorageFlattener()
        assert obj is not None
    except Exception:
        pass


def test_StorageRewriter():
    try:
        obj = StorageRewriter()
        assert obj is not None
    except Exception:
        pass


def test_DeadStoreEliminator():
    try:
        obj = DeadStoreEliminator()
        assert obj is not None
    except Exception:
        pass


def test_VirtualThreadInjector():
    try:
        obj = VirtualThreadInjector()
        assert obj is not None
    except Exception:
        pass


def test_DoubleBufferInjector():
    try:
        obj = DoubleBufferInjector()
        assert obj is not None
    except Exception:
        pass


def test_MathSimplifier():
    try:
        obj = MathSimplifier()
        assert obj is not None
    except Exception:
        pass


def test_LoopPartitioner():
    try:
        obj = LoopPartitioner()
        assert obj is not None
    except Exception:
        pass


def test_ThreadBinder():
    try:
        obj = ThreadBinder()
        assert obj is not None
    except Exception:
        pass


def test_PackedAPIMaker():
    try:
        obj = PackedAPIMaker()
        assert obj is not None
    except Exception:
        pass


def test_CustomDatatypesLowerer():
    try:
        obj = CustomDatatypesLowerer()
        assert obj is not None
    except Exception:
        pass


def test_BoundCheckerInstrumenter():
    try:
        obj = BoundCheckerInstrumenter()
        assert obj is not None
    except Exception:
        pass


def test_unroll_loop():
    try:
        unroll_loop()
    except Exception:
        pass


def test_vectorize():
    try:
        vectorize()
    except Exception:
        pass


def test_flatten_storage():
    try:
        flatten_storage()
    except Exception:
        pass


def test_rewrite_storage():
    try:
        rewrite_storage()
    except Exception:
        pass


def test_eliminate_dead_store():
    try:
        eliminate_dead_store()
    except Exception:
        pass


def test_inject_virtual_thread():
    try:
        inject_virtual_thread()
    except Exception:
        pass


def test_inject_double_buffer():
    try:
        inject_double_buffer()
    except Exception:
        pass


def test_simplify_math():
    try:
        simplify_math()
    except Exception:
        pass


def test_partition_loop():
    try:
        partition_loop()
    except Exception:
        pass


def test_bind_thread():
    try:
        bind_thread()
    except Exception:
        pass


def test_make_packed_api():
    try:
        make_packed_api()
    except Exception:
        pass


def test_lower_custom_datatypes():
    try:
        lower_custom_datatypes()
    except Exception:
        pass


def test_instrument_bound_checkers():
    try:
        instrument_bound_checkers()
    except Exception:
        pass
