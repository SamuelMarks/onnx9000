from onnx9000_tflite_converter import run


def test_run():
    assert run() == "[tflite-converter] processed"
