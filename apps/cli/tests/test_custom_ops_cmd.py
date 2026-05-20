import subprocess
import unittest


class TestCustomOpsCmd(unittest.TestCase):
    def test_custom_ops_cmd(self):
        res = subprocess.run(
            ["uv", "run", "onnx9000", "custom-ops", "fake_ops.json"], capture_output=True, text=True
        )
        self.assertIn("Registering custom ops", res.stdout)


if __name__ == "__main__":
    unittest.main()
