from onnx9000_profiler import Profiler


def test_profiler():
    p = Profiler("test.onnx")
    assert p.get_peak_memory() == 0.0
    p.run()
    assert p.get_peak_memory() == 42.5
