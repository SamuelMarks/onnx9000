"""Custom ops package."""


class CustomOpRegistry:
    """Registry for custom ONNX operations."""

    def __init__(self):
        self._ops = {}

    def register(self, op_name: str, op_func: callable):
        """Register a custom operation."""
        self._ops[op_name] = op_func

    def get_op(self, op_name: str) -> callable:
        """Get a registered operation."""
        return self._ops.get(op_name)

    def list_ops(self) -> list:
        """List all registered operation names."""
        return list(self._ops.keys())


# Global registry instance
registry = CustomOpRegistry()
