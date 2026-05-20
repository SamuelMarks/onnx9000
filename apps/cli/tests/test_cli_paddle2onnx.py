import subprocess
import unittest


class TestCliPaddle2ONNX(unittest.TestCase):
    def test_paddle2onnx(self):
        res = subprocess.run(
            ["uv", "run", "onnx9000", "paddle2onnx", "model"], capture_output=True, text=True
        )
        self.assertIn("Paddle2ONNX processed", res.stdout)


if __name__ == "__main__":
    unittest.main()
