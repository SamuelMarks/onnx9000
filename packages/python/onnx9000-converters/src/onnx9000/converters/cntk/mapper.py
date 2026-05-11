"""CNTK to ONNX IR Mapper."""

from typing import Any

from onnx9000.core.ir import Graph, Node, Variable


class CNTKMapper:
    """Mapper to convert CNTK parsed models to ONNX IR."""

    def __init__(self, model_info: dict[str, Any]):
        """Initialize mapper."""
        self.model_info = model_info
        self.graph = Graph("CNTKModel")
        self.tensors = {}

    def get_tensor(self, name: str) -> Variable:
        """Get or create tensor."""
        if name not in self.tensors:
            t = Variable(name)
            self.tensors[name] = t
            self.graph.add_tensor(t)
        return self.tensors[name]

    def map(self) -> Graph:
        """Map CNTK model to ONNX IR."""
        for inp in self.model_info.get("inputs", []):
            t = self.get_tensor(inp["name"])
            self.graph.inputs.append(t)

        for node_info in self.model_info.get("nodes", []):
            op_type = node_info.get("op", "")
            name = node_info.get("name", "")
            inputs = [self.get_tensor(i) for i in node_info.get("inputs", [])]
            outputs = [self.get_tensor(o) for o in node_info.get("outputs", [])]

            # Simple handling of dynamic sequence axes
            # CNTK often has dynamic sequence axes which ONNX represents using dynamic dimensions (-1)
            # or Sequence constructs. We map them simply.
            if op_type == "Convolution":
                node = Node("Conv", inputs=inputs, outputs=outputs, name=name)
                self.graph.add_node(node)
            elif op_type == "Plus":
                node = Node("Add", inputs=inputs, outputs=outputs, name=name)
                self.graph.add_node(node)
            else:
                node = Node(
                    op_type if op_type else "Identity", inputs=inputs, outputs=outputs, name=name
                )
                self.graph.add_node(node)

        for out in self.model_info.get("outputs", []):
            t = self.get_tensor(out["name"])
            if t not in self.graph.outputs:
                self.graph.outputs.append(t)

        return self.graph
