import subprocess
import unittest


class TestMissingSdkCmd(unittest.TestCase):
    def run_cmd(self, *args):
        return subprocess.run(["uv", "run", "onnx9000", *args], capture_output=True, text=True)

    def test_mobile(self):
        res = self.run_cmd("mobile-memory", "test")
        self.assertIn("Mobile Memory processed", res.stdout)

    def test_prog(self):
        res = self.run_cmd("progressive-loading", "test")
        self.assertIn("Progressive Loading processed", res.stdout)

    def test_arch(self):
        res = self.run_cmd("new-model-arch", "test")
        self.assertIn("New Model Arch processed", res.stdout)

    def test_zero(self):
        res = self.run_cmd("zero-dep-classifier", "test")
        self.assertIn("Zero Dep Classifier processed", res.stdout)


if __name__ == "__main__":
    unittest.main()
