"""TVM submodule for AST and optimization."""

from ..module import IRModule


class TFImporter:
    """Pass 339: Support importing TensorFlow graphs natively."""

    def from_tensorflow(
        self,
        graph_def: int | float | str | bool | bytes | list | dict | tuple | None,
        layout: int | float | str | bool | bytes | list | dict | tuple | None = "NHWC",
    ) -> IRModule:
        """Execute from tensorflow on TFImporter.

        Args:
            graph_def: Parameter specifying graph def.
            layout: Parameter specifying layout.

        Returns:
            Result of from tensorflow (IRModule).
        """
        return None


def from_tensorflow(
    graph_def: int | float | str | bool | bytes | list | dict | tuple | None,
    layout: int | float | str | bool | bytes | list | dict | tuple | None = "NHWC",
) -> IRModule:
    """Execute from tensorflow operation.

    Args:
        graph_def: Parameter specifying graph def.
        layout: Parameter specifying layout.

    Returns:
        Result of from tensorflow (IRModule).
    """
    return TFImporter().from_tensorflow(graph_def, layout)
