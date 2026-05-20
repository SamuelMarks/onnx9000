import subprocess
import unittest


class TestExtendedConvertersCmd(unittest.TestCase):
    def run_cmd(self, fmt):
        return subprocess.run(
            ["uv", "run", "onnx9000", "convert", "--from", fmt, "fake_model"],
            capture_output=True,
            text=True,
        )

    def test_paddle(self):
        res = self.run_cmd("paddle")
        self.assertIn("Converting from paddle", res.stdout)

    def test_keras(self):
        res = self.run_cmd("keras")
        self.assertIn("Converting from keras", res.stdout)

    def test_sklearn(self):
        res = self.run_cmd("sklearn")
        self.assertIn("Converting from sklearn", res.stdout)


if __name__ == "__main__":
    unittest.main()
