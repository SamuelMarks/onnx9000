import subprocess
import unittest


class TestCliKeras2ONNX(unittest.TestCase):
    def test_keras2onnx(self):
        res = subprocess.run(
            ["uv", "run", "onnx9000", "keras2onnx", "model"], capture_output=True, text=True
        )
        self.assertIn("Keras2ONNX processed", res.stdout)


if __name__ == "__main__":
    unittest.main()
