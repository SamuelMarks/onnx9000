"""TVM submodule for AST and optimization."""

from ..module import IRModule


class TFImporter:
    """Pass 339: Support importing TensorFlow graphs natively."""

    def from_tensorflow(self, graph_def, layout="NHWC") -> IRModule:
        """Do the function."""
        pass


def from_tensorflow(graph_def, layout="NHWC") -> IRModule:
    """Do the function."""
    return TFImporter().from_tensorflow(graph_def, layout)
