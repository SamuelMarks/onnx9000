from onnx9000_sparse import run


def test_run():
    assert run() == "[sparse] processed"
