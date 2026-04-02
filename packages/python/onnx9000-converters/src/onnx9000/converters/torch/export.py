"""torch.export parser for onnx9000."""

import torch
from onnx9000.converters.frontend.builder import GraphBuilder
from onnx9000.converters.torch.fx import FXParser


class ExportParser(FXParser):
    """Parser for torch.export.ExportedProgram."""

    def __init__(self, ep: "torch.export.ExportedProgram") -> None:
        """Initialize the Export parser.

        Args:
            ep: The ExportedProgram to parse.
        """
        super().__init__(ep.graph_module)
        self.ep = ep

    def parse(self) -> GraphBuilder:
        """Parse the ExportedProgram into a GraphBuilder.

        Returns:
            The populated GraphBuilder.
        """
        # ExportedProgram has more structured info about inputs/parameters
        # but FXParser.parse() already walks the GraphModule nodes.
        # We might need to adjust placeholder handling to match ep.graph_signature.
        return super().parse()
