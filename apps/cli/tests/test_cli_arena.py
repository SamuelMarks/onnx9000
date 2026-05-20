import subprocess
import unittest


class TestCliArena(unittest.TestCase):
    def test_arena(self):
        res = subprocess.run(
            ["uv", "run", "onnx9000", "arena", "model"], capture_output=True, text=True
        )
        self.assertIn("Arena processed", res.stdout)


if __name__ == "__main__":
    unittest.main()
