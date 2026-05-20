import subprocess
import unittest


class TestSpecOrphansCmd(unittest.TestCase):
    def run_cmd(self, *args):
        return subprocess.run(["uv", "run", "onnx9000", *args], capture_output=True, text=True)

    def test_ort(self):
        res = self.run_cmd("ort-training", "test")
        self.assertIn("ORT Training processed", res.stdout)

    def test_olive(self):
        res = self.run_cmd("olive-optimizer", "test")
        self.assertIn("Olive Optimizer processed", res.stdout)

    def test_triton(self):
        res = self.run_cmd("triton-server", "test")
        self.assertIn("Triton Server processed", res.stdout)

    def test_onnx_tool(self):
        res = self.run_cmd("onnx-tool", "test")
        self.assertIn("ONNX Tool processed", res.stdout)


if __name__ == "__main__":
    unittest.main()
