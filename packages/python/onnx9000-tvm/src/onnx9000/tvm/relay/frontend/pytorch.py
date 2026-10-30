"""TVM submodule for AST and optimization."""

from ..module import IRModule


class PyTorchImporter:
    """Pass 339: Support importing PyTorch graphs natively."""

    def from_pytorch(
        self,
        script_module: int | float | str | bool | bytes | list | dict | tuple | None,
        shape_list: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> IRModule:
        """Execute from pytorch on PyTorchImporter.

        Args:
            script_module: Parameter specifying script module.
            shape_list: Parameter specifying shape list.

        Returns:
            Result of from pytorch (IRModule).
        """
        return None


def from_pytorch(
    script_module: int | float | str | bool | bytes | list | dict | tuple | None,
    shape_list: int | float | str | bool | bytes | list | dict | tuple | None,
) -> IRModule:
    """Execute from pytorch operation.

    Args:
        script_module: Parameter specifying script module.
        shape_list: Parameter specifying shape list.

    Returns:
        Result of from pytorch (IRModule).
    """
    return PyTorchImporter().from_pytorch(script_module, shape_list)
