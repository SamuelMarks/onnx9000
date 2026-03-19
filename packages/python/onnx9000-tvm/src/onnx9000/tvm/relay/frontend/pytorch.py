"""TVM submodule for AST and optimization."""

from ..module import IRModule


class PyTorchImporter:
    """Pass 339: Support importing PyTorch graphs natively."""

    def from_pytorch(self, script_module, shape_list) -> IRModule:
        """Do the function."""
        pass


def from_pytorch(script_module, shape_list) -> IRModule:
    """Do the function."""
    return PyTorchImporter().from_pytorch(script_module, shape_list)
