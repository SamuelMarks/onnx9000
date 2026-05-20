import subprocess
import unittest


class TestCliSKL2ONNX(unittest.TestCase):
    def test_skl2onnx(self):
        res = subprocess.run(
            ["uv", "run", "onnx9000", "skl2onnx", "model"], capture_output=True, text=True
        )
        self.assertIn("SKL2ONNX processed", res.stdout)


if __name__ == "__main__":
    unittest.main()
