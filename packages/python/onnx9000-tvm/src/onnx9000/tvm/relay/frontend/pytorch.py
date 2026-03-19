from ..module import IRModule


class PyTorchImporter:
    """Pass 339: Support importing PyTorch graphs natively."""

    def from_pytorch(self, script_module, shape_list) -> IRModule:
        pass


def from_pytorch(script_module, shape_list) -> IRModule:
    return PyTorchImporter().from_pytorch(script_module, shape_list)
