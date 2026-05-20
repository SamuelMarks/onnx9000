import unittest

from onnx9000_custom_ops import registry


class TestCustomOps(unittest.TestCase):
    def test_registry(self):
        def my_op():
            return "ok"

        registry.register("MyOp", my_op)
        self.assertIn("MyOp", registry.list_ops())
        self.assertEqual(registry.get_op("MyOp")(), "ok")


if __name__ == "__main__":
    unittest.main()
