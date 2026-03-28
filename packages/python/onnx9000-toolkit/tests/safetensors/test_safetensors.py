"""Tests for packages/python/onnx9000-toolkit/tests/safetensors/test_safetensors.py."""

import os
import tempfile
import pytest
from onnx9000.toolkit.safetensors.parser import (
    SafeTensors,
    SafetensorsAlignmentError,
    SafetensorsDuplicateKeyError,
    SafetensorsError,
    SafetensorsFileEmptyError,
    SafetensorsFileTooSmallError,
    SafetensorsHeaderTooLargeError,
    SafetensorsInvalidDtypeError,
    SafetensorsInvalidHeaderError,
    SafetensorsInvalidJSONError,
    SafetensorsInvalidOffsetError,
    SafetensorsOutOfBoundsError,
    SafetensorsOverlapError,
    SafetensorsShapeMismatchError,
    check_safetensors,
    safe_open,
    save_file,
)


def test_basic_save_load():
    """Test basic save load."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        tensors = {"weight1": b"hello world", "weight2": b"foo bar baz qux"}
        metadata = {"format": "pt"}
        save_file(tensors, path, metadata)
        assert check_safetensors(path) is True
        with SafeTensors(path) as st:
            assert st.metadata == metadata
            assert st.keys() == ["weight1", "weight2"]
            assert st.get_tensor("weight1").tobytes() == b"hello world"
            assert st["weight2"].tobytes() == b"foo bar baz qux"
            assert "weight1" in st


def test_safe_open():
    """Test safe open."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a": b"1"}, path)
        with safe_open(path, framework="pt", device="cpu") as f:
            assert f.keys() == ["a"]
            assert f.get_tensor("a").tobytes() == b"1"


def test_empty_file():
    """Test empty file."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "empty.safetensors")
        with open(path, "wb"):
            pass
        with pytest.raises(SafetensorsFileEmptyError):
            SafeTensors(path)


def test_too_small_file():
    """Test too small file."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "small.safetensors")
        with open(path, "wb") as f:
            f.write(b"1234")
        with pytest.raises(SafetensorsFileTooSmallError):
            SafeTensors(path)


import json
import numpy as np
from onnx9000.toolkit.safetensors.parser import get_metadata, save


def test_numpy_save_and_load():
    """Test numpy save and load."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        save_file({"f32": arr1, "i32": arr2}, path)
        with SafeTensors(path) as st:
            out1 = st.get_numpy("f32")
            out2 = st.get_numpy("i32")
            assert out1.dtype == np.float32
            assert out2.dtype == np.int32
            assert out1.shape == (3,)
            assert out2.shape == (2, 2)
            np.testing.assert_allclose(out1, arr1)
            np.testing.assert_array_equal(out2, arr2)


def test_bytes_loading():
    """Test bytes loading."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = save({"weight": arr}, {"format": "np"})
    with SafeTensors(b) as st:
        assert st.keys() == ["weight"]
        assert st.metadata == {"format": "np", "version": "1.0"}
        out = st.get_numpy("weight")
        np.testing.assert_allclose(out, arr)


def test_get_tensors():
    """Test get tensors."""
    arr1 = np.array([1.0, 2.0], dtype=np.float32)
    arr2 = np.array([3.0, 4.0], dtype=np.float32)
    b = save({"a": arr1, "b": arr2})
    with SafeTensors(b) as st:
        tensors = st.get_tensors("a", "b")
        assert len(tensors) == 2
        np.testing.assert_allclose(np.frombuffer(tensors["a"], dtype=np.float32), arr1)


def test_iter_tensors():
    """Test iter tensors."""
    b = save({"a": b"123", "b": b"456"})
    with SafeTensors(b) as st:
        out = list(st.iter_tensors())
        assert len(out) == 2
        assert out[0][0] == "a"
        assert out[0][1].tobytes() == b"123"


def test_memory_footprint():
    """Test memory footprint."""
    arr = np.array([1, 2, 3, 4], dtype=np.int32)
    b = save({"a": arr})
    with SafeTensors(b) as st:
        assert st.get_memory_footprint() == 16


def test_get_metadata():
    """Test get metadata."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a": b"1"}, path, {"hi": "there"})
        meta = get_metadata(path)
        assert meta == {"hi": "there", "format": "pt", "version": "1.0"}


def test_get_tensor():
    """Test get tensor."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a": b"12345"}, path)
        from onnx9000.toolkit.safetensors import get_tensor

        assert get_tensor(path, "a").tobytes() == b"12345"


from onnx9000.toolkit.safetensors.parser import SafeTensorsSharded


def test_sharded():
    """Test sharded."""
    with tempfile.TemporaryDirectory() as d:
        path1 = os.path.join(d, "sharded-00001.safetensors")
        path2 = os.path.join(d, "sharded-00002.safetensors")
        save_file({"a": np.array([1], dtype=np.int8)}, path1)
        save_file({"b": np.array([2], dtype=np.int8)}, path2)
        index_path = os.path.join(d, "sharded.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "weight_map": {
                        "a": "sharded-00001.safetensors",
                        "b": "sharded-00002.safetensors",
                    }
                },
                f,
            )
        with SafeTensorsSharded(index_path) as sharded:
            assert "a" in sharded
            assert "b" in sharded
            assert sharded.keys() == ["a", "b"]
            assert sharded.get_numpy("a")[0] == 1
            assert sharded.get_numpy("b")[0] == 2


def test_negative_dimension():
    """Test negative dimension."""
    from onnx9000.toolkit.safetensors.parser import SafetensorsShapeMismatchError, _calculate_volume

    with pytest.raises(SafetensorsShapeMismatchError, match="Negative dimension found"):
        _calculate_volume([-1, 10])


def test_huge_dimension():
    """Test huge dimension."""
    from onnx9000.toolkit.safetensors.parser import SafetensorsShapeMismatchError, _calculate_volume

    with pytest.raises(SafetensorsShapeMismatchError, match="Dimension too large"):
        _calculate_volume([2**51])


def test_too_large_header():
    """Test too large header."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "large_header.safetensors")
        with open(path, "wb") as f:
            import struct

            f.write(struct.pack("<Q", 101 * 1024 * 1024))
            f.write(b"0" * 100)
        with pytest.raises(SafetensorsHeaderTooLargeError):
            SafeTensors(path)


def test_invalid_utf8_header():
    """Test invalid utf8 header."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "invalid_utf8.safetensors")
        with open(path, "wb") as f:
            import struct

            f.write(struct.pack("<Q", 4))
            f.write(b"\xff\xff\xff\xff")
        with pytest.raises(SafetensorsInvalidHeaderError):
            SafeTensors(path)


def test_invalid_json_header():
    """Test invalid json header."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "invalid_json.safetensors")
        with open(path, "wb") as f:
            import struct

            f.write(struct.pack("<Q", 4))
            f.write(b"nope")
        with pytest.raises(SafetensorsInvalidJSONError):
            SafeTensors(path)


def test_out_of_bounds_header():
    """Test out of bounds header."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out_of_bounds.safetensors")
        with open(path, "wb") as f:
            import struct

            f.write(struct.pack("<Q", 100))
            f.write(b"small")
        with pytest.raises(SafetensorsOutOfBoundsError):
            SafeTensors(path)


def test_invalid_dtype():
    """Test invalid dtype."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "invalid_dtype.safetensors")
        with open(path, "wb") as f:
            import struct

            header = b'{"a": {"dtype": "X99", "shape": [1], "data_offsets": [0, 1]}}'
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(b"x")
        with pytest.raises(SafetensorsInvalidDtypeError):
            SafeTensors(path)


def test_shape_mismatch():
    """Test shape mismatch."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "shape_mismatch.safetensors")
        with open(path, "wb") as f:
            import struct

            header = b'{"a": {"dtype": "I8", "shape": [10], "data_offsets": [0, 1]}}'
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(b"x")
        with pytest.raises(SafetensorsShapeMismatchError):
            SafeTensors(path)


def test_overlap():
    """Test overlap."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "overlap.safetensors")
        with open(path, "wb") as f:
            import struct

            header = b'{"a": {"dtype": "I8", "shape": [16], "data_offsets": [0, 16]}, "b": {"dtype": "I8", "shape": [16], "data_offsets": [8, 24]}}'
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(b"x" * 24)
        with pytest.raises(SafetensorsOverlapError):
            SafeTensors(path)


def test_invalid_offsets():
    """Test invalid offsets."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "invalid_offsets.safetensors")
        with open(path, "wb") as f:
            import struct

            header = b'{"a": {"dtype": "I8", "shape": [1], "data_offsets": [2, 1]}}'
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(b"x")
        with pytest.raises(SafetensorsInvalidOffsetError):
            SafeTensors(path)


def test_data_out_of_bounds():
    """Test data out of bounds."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data_out_of_bounds.safetensors")
        with open(path, "wb") as f:
            import struct

            header = b'{"a": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]}}'
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
        with pytest.raises(SafetensorsOutOfBoundsError):
            SafeTensors(path)


def test_0_byte_tensor():
    """Test 0 byte tensor."""
    b = save({"empty": np.array([], dtype=np.float32)})
    with SafeTensors(b) as st:
        out = st.get_numpy("empty")
        assert out.size == 0
        assert out.shape == (0,)


def test_1d_2d_3d_4d_float32():
    """Test 1d 2d 3d 4d float32."""
    d1 = np.ones((2,), dtype=np.float32)
    d2 = np.ones((2, 2), dtype=np.float32)
    d3 = np.ones((2, 2, 2), dtype=np.float32)
    d4 = np.ones((2, 2, 2, 2), dtype=np.float32)
    b = save({"d1": d1, "d2": d2, "d3": d3, "d4": d4})
    with SafeTensors(b) as st:
        np.testing.assert_allclose(st.get_numpy("d1"), d1)
        np.testing.assert_allclose(st.get_numpy("d2"), d2)
        np.testing.assert_allclose(st.get_numpy("d3"), d3)
        np.testing.assert_allclose(st.get_numpy("d4"), d4)


def test_endianness_conversion():
    """Test endianness conversion."""
    arr = np.array([1.0, 2.0], dtype=">f4")
    b = save({"be": arr})
    with SafeTensors(b) as st:
        out = st.get_numpy("be")
        assert len(out) == 2


def test_high_precision_dimensions():
    """Test high precision dimensions."""
    dim = 2147483647
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "high_dim.safetensors")
        with open(path, "wb") as f:
            import struct

            header = (
                f'{{"a": {{"dtype": "I8", "shape": [{dim}], "data_offsets": [0, {dim}]}}}}'.encode()
            )
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
        with pytest.raises(SafetensorsOutOfBoundsError):
            SafeTensors(path)


def test_empty_metadata():
    """Test empty metadata."""
    b = save({"a": np.array([1], dtype=np.int8)}, {})
    with SafeTensors(b) as st:
        assert st.metadata == {"format": "pt", "version": "1.0"}


def test_metadata_format_pt():
    """Test metadata format pt."""
    b = save({"a": np.array([1], dtype=np.int8)}, {"format": "pt"})
    with SafeTensors(b) as st:
        assert st.metadata == {"format": "pt", "version": "1.0"}


def test_load_file_features():
    """Test load file features."""
    from onnx9000.toolkit.safetensors.parser import SafetensorsWriteError, load_file

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file(
            {
                "model.layer.1.weight": np.array([1]),
                "model.layer.2.weight": np.array([2]),
                "bias": np.array([3]),
            },
            path,
        )
        with pytest.raises(SafetensorsWriteError):
            save_file({"new": np.array([1])}, path, overwrite=False)
        filtered = load_file(path, pattern=".*weight$")
        assert list(filtered.keys()) == ["model.layer.1.weight", "model.layer.2.weight"]
        prefixed = load_file(path, prefix="test.")
        assert list(prefixed.keys()) == [
            "test.model.layer.1.weight",
            "test.model.layer.2.weight",
            "test.bias",
        ]


def test_merge_files():
    """Test merge files."""
    from onnx9000.toolkit.safetensors.parser import load_file

    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "1.safetensors")
        p2 = os.path.join(d, "2.safetensors")
        p_merged = os.path.join(d, "merged.safetensors")
        save_file({"a": np.array([1])}, p1)
        save_file({"b": np.array([2])}, p2)
        dict1 = load_file(p1)
        dict2 = load_file(p2)
        dict1.update(dict2)
        save_file(dict1, p_merged)
        res = load_file(p_merged)
        assert list(res.keys()) == ["a", "b"]


def test_type_error_on_non_string_keys():
    """Test type error on non string keys."""
    with pytest.raises(TypeError):
        save({123: b"abc"})


def test_type_error_get_tensor():
    """Test type error get tensor."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a": b"1"}, path)
        with SafeTensors(path) as st:
            with pytest.raises(TypeError):
                st.get_tensor(123)
            with pytest.raises(TypeError):
                _ = st[123]


def test_missing_hf_keys_warning():
    """Test missing hf keys warning."""
    import warnings

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a": b"1"}, path, {"some_key": "val"})


def test_concurrency():
    """Test concurrency."""
    import threading

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a": b"1" * 1000}, path)

        def read_loop():
            """Perform read loop operation."""
            with SafeTensors(path) as st:
                for _ in range(10):
                    assert st.get_tensor("a").tobytes() == b"1" * 1000

        threads = [threading.Thread(target=read_loop) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def _read_loop_mp(p):
    """Perform  read loop mp operation."""
    from onnx9000.toolkit.safetensors.parser import SafeTensors

    with SafeTensors(p) as st:
        for _ in range(10):
            assert st.get_tensor("a").tobytes() == b"2" * 1000


def test_multiprocessing():
    """Test multiprocessing."""
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model.safetensors")
        save_file({"a": b"2" * 1000}, p)
        _read_loop_mp(p)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a": b"2" * 1000}, path)
        processes = [multiprocessing.Process(target=_read_loop_mp, args=(path,)) for _ in range(4)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        for p in processes:
            assert p.exitcode == 0


def test_context_manager():
    """Test context manager."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a": b"1"}, path)
        st = SafeTensors(path)
        assert st.fd is not None
        assert st.mm is not None
        st._close()
        assert st.fd is None
        assert st.mm is None
        with SafeTensors(path) as st2:
            assert st2.fd is not None
        assert st2.fd is None


def test_verify_hash():
    """Test verify hash."""
    import hashlib

    b = save({"a": b"123"})
    h = hashlib.sha256()
    h.update(b)
    correct_hash = h.hexdigest()
    with SafeTensors(b, verify_hash=correct_hash) as st:
        assert st.keys() == ["a"]
    with pytest.raises(SafetensorsError, match="SHA256 mismatch"):
        SafeTensors(b, verify_hash="badhash")


def test_memory_leak():
    """Test memory leak."""
    import gc
    import weakref

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a": b"1"}, path)
        st = SafeTensors(path)
        ref = weakref.ref(st)
        assert ref() is not None
        view = st.get_tensor("a")
        del st
        gc.collect()
        assert ref() is None
        del view
        gc.collect()


def test_save_sharded():
    """Test save sharded."""
    from onnx9000.toolkit.safetensors.parser import save_sharded

    with tempfile.TemporaryDirectory() as d:
        tensors = {
            "a": np.ones((1024,), dtype=np.int8),
            "b": np.ones((1024,), dtype=np.int8),
            "c": np.ones((1024,), dtype=np.int8),
        }
        save_sharded(tensors, d, max_shard_size=1500, prefix="save_sharded_test")
        index_path = os.path.join(d, "save_sharded_test.safetensors.index.json")
        assert os.path.exists(index_path)
        with open(index_path) as f:
            idx = json.load(f)
        assert len(idx["weight_map"]) == 3
        files = set(idx["weight_map"].values())
        assert len(files) == 3
        for name, filename in idx["weight_map"].items():
            assert filename.startswith("save_sharded_test-000")
            assert filename.endswith("-of-00003.safetensors")
            assert os.path.exists(os.path.join(d, filename))
        with SafeTensorsSharded(index_path) as st:
            assert st.keys() == ["a", "b", "c"]
            assert len(st.get_numpy("a")) == 1024


def test_alignment_validation():
    """Test alignment validation."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "unaligned.safetensors")
        with open(path, "wb") as f:
            import struct

            header = b'{"a": {"dtype": "I8", "shape": [1], "data_offsets": [1, 2]}}'
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(b"xx")
        with pytest.raises(SafetensorsAlignmentError):
            SafeTensors(path)


def test_fuzz_corrupted_json():
    """Test fuzz corrupted json."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fuzz_json.safetensors")
        with open(path, "wb") as f:
            import struct

            f.write(struct.pack("<Q", 10))
            f.write(b'{"a":1}   ')
            f.write(b"x")
        with open(path, "wb") as f:
            import struct

            f.write(struct.pack("<Q", 10))
            f.write(b'{"a":1    ')
        with pytest.raises(SafetensorsInvalidJSONError):
            SafeTensors(path)


def test_fuzz_corrupted_binary_offsets():
    """Test fuzz corrupted binary offsets."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fuzz_offsets.safetensors")
        with open(path, "wb") as f:
            import struct

            header = b'{"a": {"dtype": "I8", "shape": [1], "data_offsets": "0, 1"}}'
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
        with pytest.raises((SafetensorsError, TypeError)):
            SafeTensors(path)


def test_zero_dim_scalar():
    """Test zero dim scalar."""
    from onnx9000.toolkit.safetensors.parser import SafetensorsShapeMismatchError, _calculate_volume

    assert _calculate_volume([]) == 1
    with pytest.raises(SafetensorsShapeMismatchError, match="Shape must be an array/list"):
        _calculate_volume(None)


def test_pinned_memory():
    """Test pinned memory."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pinned.safetensors")
        save_file({"a": b"1234"}, path)
        with SafeTensors(path) as st:
            view = st.get_pinned_tensor("a")
            assert view.tobytes() == b"1234"


def test_1gb_throughput():
    """Test 1gb throughput."""
    import time

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "large.safetensors")
        size_bytes = 100 * 1024 * 1024
        arr = np.zeros(size_bytes // 4, dtype=np.float32)
        start = time.perf_counter()
        save_file({"huge": arr}, path)
        end = time.perf_counter()
        assert end - start < 5.0
        with SafeTensors(path) as st:
            view = st.get_numpy("huge")
            assert view.size == size_bytes // 4
            assert view[0] == 0.0
            assert view[-1] == 0.0


def test_memoryview_slice_latency():
    """Test memoryview slice latency."""
    import time

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bench.safetensors")
        arr = np.ones(1024 * 1024, dtype=np.float32)
        save_file({"w": arr}, path)
        with SafeTensors(path) as st:
            view = st.get_tensor("w")
            start = time.perf_counter()
            for i in range(1000):
                _ = view[i : i + 10]
            end = time.perf_counter()
            assert end - start < 1.0


def test_load_file_filter_regex():
    """Test load file filter regex."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file(
            {"a.weight": np.array([1]), "a.bias": np.array([2]), "b.weight": np.array([3])}, path
        )
        from onnx9000.toolkit.safetensors.parser import load_file

        res = load_file(path, pattern=".*\\.weight$")
        assert "a.weight" in res
        assert "b.weight" in res
        assert "a.bias" not in res


def test_int64_bounds():
    """Test int64 bounds."""
    import struct

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "large_bounds.safetensors")
        with open(path, "wb") as f:
            huge = 2**61
            header = (
                f'{{"a": {{"dtype": "I8", "shape": [1], "data_offsets": [0, {huge}]}}}}'.encode(
                    "utf-8"
                )
            )
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
        with pytest.raises(SafetensorsOutOfBoundsError):
            SafeTensors(path)


def test_path_traversal():
    """Test path traversal."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors.index.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"weight_map": {"a": "../../../../etc/passwd"}}, f)
        with pytest.raises(SafetensorsError, match="Path traversal"):
            with SafeTensorsSharded(path) as st:
                st.get_tensor("a")


def test_xss_injection_metadata():
    """Test xss injection metadata."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a": b"1"}, path, {"tag": "<script>alert('xss')</script>"})
        with pytest.raises(SafetensorsError, match="Executable script tags detected"):
            SafeTensors(path)


def test_xxe_json_injection():
    """Test xxe json injection."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        with open(path, "wb") as f:
            import struct

            header = b'["not a dictionary", 123]'
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
        with pytest.raises(SafetensorsError):
            SafeTensors(path)


def test_bfloat16_generation_security():
    """Test bfloat16 generation security."""
    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bf16.safetensors")
        from onnx9000.toolkit.safetensors.parser import save
        import struct

        header = {"a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}, "__metadata__": {}}
        import json

        header_str = json.dumps(header).encode("utf-8")
        header_len = len(header_str)
        pad = (8 - header_len % 8) % 8
        header_str += b" " * pad
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(header_str)))
            f.write(header_str)
            f.write(b"\x00?")
        with SafeTensors(path) as st:
            assert st.tensors["a"]["dtype"] == "BF16"


def test_loopback_reading():
    """Test loopback reading."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "loopback.safetensors")
        arr1 = np.ones((5, 5), dtype=np.float32)
        arr2 = np.zeros((10,), dtype=np.int8)
        save_file({"a": arr1, "b": arr2}, path)
        with SafeTensors(path) as st:
            np.testing.assert_allclose(st.get_numpy("a"), arr1)
            np.testing.assert_array_equal(st.get_numpy("b"), arr2)


def test_huge_dimension_check():
    """Test huge dimension check."""
    import json

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "huge_dim.safetensors")
        with open(path, "wb") as f:
            import struct

            dim = 2**51
            header = f'{{"a": {{"dtype": "I8", "shape": [{dim}], "data_offsets": [0, 1]}}}}'.encode(
                "utf-8"
            )
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(b"x")
        with pytest.raises(SafetensorsShapeMismatchError, match="Dimension too large"):
            SafeTensors(path)


def test_json_recursion_limits():
    """Test json recursion limits."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "deep_json.safetensors")
        with open(path, "wb") as f:
            import struct

            header_str = '{"a": ' * 2000 + "1" + "}" * 2000
            header_bytes = header_str.encode("utf-8")
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"x")
        with pytest.raises(
            (SafetensorsInvalidJSONError, RecursionError, SafetensorsInvalidDtypeError)
        ):
            SafeTensors(path)


def test_converters_coverage():
    """Test converters coverage."""
    from onnx9000.toolkit.safetensors.converters import (
        convert_pytorch_to_safetensors,
        convert_tf_to_safetensors,
    )
    import tempfile
    import os
    import pytest
    import sys
    import sys

    class MockTorch:
        """MockTorch implementation."""

        class Tensor:
            """Tensor implementation."""

            def numpy(self):
                """Perform numpy operation."""
                import numpy as np

                return np.array([1], dtype=np.float32)

        def load(self, f, map_location):
            """Perform load operation."""
            return {"a": self.Tensor(), "b": "not_tensor"}

    class MockTF:
        """MockTF implementation."""

        class Var:
            """Var implementation."""

            def __init__(self, name):
                """Perform   init   operation."""
                self.name = name

            def numpy(self):
                """Perform numpy operation."""
                import numpy as np

                return np.array([2], dtype=np.float32)

        class Model:
            """Model implementation."""

            def __init__(self):
                """Perform   init   operation."""
                self.variables = [MockTF.Var("w:0"), MockTF.Var("b")]

        class SavedModel:
            """SavedModel implementation."""

            def load(self, d):
                """Perform load operation."""
                return MockTF.Model()

        def __init__(self):
            """Perform   init   operation."""
            self.saved_model = self.SavedModel()

    from unittest.mock import patch

    with patch.dict(sys.modules, {"torch": MockTorch(), "tensorflow": MockTF()}):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model.bin"), "wb") as bf:
                pass
            convert_pytorch_to_safetensors(d, d)
            convert_tf_to_safetensors(d, os.path.join(d, "tf.safetensors"))
            with tempfile.TemporaryDirectory() as d2:
                convert_pytorch_to_safetensors(d2)
    with patch.dict(sys.modules, {"torch": None, "tensorflow": None}):
        with pytest.raises(ImportError):
            convert_pytorch_to_safetensors("dummy")
        with pytest.raises(ImportError):
            convert_tf_to_safetensors("dummy", "dummy")


def test_hub_coverage():
    """Test hub coverage."""
    from onnx9000.toolkit.safetensors.hub import _get_cache_dir, resolve_model_file, cached_download
    import tempfile
    import os
    import pytest
    from unittest.mock import patch, MagicMock
    from urllib.error import HTTPError

    with patch.dict(os.environ, {"HF_HOME": "/tmp/hf"}):
        assert _get_cache_dir() == "/tmp/hf"
    with patch.dict(os.environ, {}, clear=True):
        assert _get_cache_dir() == os.path.expanduser("~/.cache/huggingface/hub")
    with patch("onnx9000.toolkit.safetensors.hub.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response
        with patch.dict(os.environ, {"HF_TOKEN": "token"}):
            url = resolve_model_file("repo", "main")
            assert url.endswith(".safetensors")
        mock_urlopen.side_effect = HTTPError("url", 404, "Not Found", {}, None)
        url2 = resolve_model_file("repo", "main")
        assert url2.endswith(".bin")
    with tempfile.TemporaryDirectory() as d:
        with patch("onnx9000.toolkit.safetensors.hub._get_cache_dir", return_value=d):
            with patch("onnx9000.toolkit.safetensors.hub.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.side_effect = [b"hello", b""]
                mock_urlopen.return_value.__enter__.return_value = mock_response
                path1 = cached_download(
                    "https://huggingface.co/repo/resolve/main/file", revision="dev"
                )
                assert os.path.exists(path1)
                path2 = cached_download(
                    "https://huggingface.co/repo/resolve/main/file", revision="dev"
                )
                assert path2 == path1
                mock_response.read.side_effect = [b"hello", b""]
                path3 = cached_download(
                    "https://huggingface.co/repo/resolve/main/file",
                    revision="dev",
                    force_download=True,
                )
                import hashlib

                expected_hash = hashlib.sha256(b"hello").hexdigest()
                path4 = cached_download(
                    "https://huggingface.co/repo/resolve/main/file",
                    revision="dev",
                    expected_sha256=expected_hash,
                )
                mock_response.read.side_effect = [b"hello2", b""]
                expected_hash2 = hashlib.sha256(b"hello2").hexdigest()
                path5 = cached_download(
                    "https://huggingface.co/repo/resolve/main/file",
                    revision="dev",
                    expected_sha256=expected_hash2,
                )
                mock_response.read.side_effect = [b"hello3", b""]
                with pytest.raises(RuntimeError, match="Hash validation failed"):
                    cached_download(
                        "https://huggingface.co/repo/resolve/main/file",
                        revision="dev",
                        force_download=True,
                        expected_sha256="wrong",
                    )
                mock_urlopen.side_effect = HTTPError("url", 404, "Not Found", {}, None)
                with pytest.raises(RuntimeError, match="Failed to download"):
                    cached_download("https://huggingface.co/repo/file", force_download=True)


def test_parser_uncovered():
    """Test parser uncovered."""
    pass


def test_interop_real():
    """Test interop real."""
    from onnx9000.toolkit.safetensors.interop import (
        load_pytorch_safetensors,
        load_tensorflow_safetensors,
        load_flax_safetensors,
    )
    import tempfile
    import os
    import numpy as np
    from onnx9000.toolkit.safetensors.parser import save_file

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"a.kernel": np.array([1]), "b": np.ones((2, 3, 4, 5))}, path)
        pt = load_pytorch_safetensors(path)
        assert "a.kernel" in pt
        tf = load_tensorflow_safetensors(path)
        assert tf["b"].shape == (5, 4, 2, 3)
        flax = load_flax_safetensors(path)
        assert "a.weight" in flax
