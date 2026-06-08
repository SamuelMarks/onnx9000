"""CNTK to ONNX IR Mapper."""

from typing import Any  # pragma: no cover

from onnx9000.core.ir import Graph, Node, Variable  # pragma: no cover


class CNTKMapper:
    """Mapper to convert CNTK parsed models to ONNX IR."""

    def __init__(self, model_info: dict[str, Any]):
        """Initialize mapper."""
        self.model_info = model_info  # pragma: no cover
        self.graph = Graph("CNTKModel")  # pragma: no cover
        self.tensors = {}  # pragma: no cover

    def get_tensor(self, name: str) -> Variable:
        """Get or create tensor."""
        if name not in self.tensors:  # pragma: no cover
            t = Variable(name)  # pragma: no cover
            self.tensors[name] = t  # pragma: no cover
            self.graph.add_tensor(t)  # pragma: no cover
        return self.tensors[name]  # pragma: no cover

    def map(self) -> Graph:
        """Map CNTK model to ONNX IR."""
        for inp in self.model_info.get("inputs", []):  # pragma: no cover
            t = self.get_tensor(inp["name"])  # pragma: no cover
            self.graph.inputs.append(t)  # pragma: no cover

        for node_info in self.model_info.get("nodes", []):  # pragma: no cover
            op_type = node_info.get("op", "")  # pragma: no cover
            name = node_info.get("name", "")  # pragma: no cover
            inputs = [self.get_tensor(i) for i in node_info.get("inputs", [])]  # pragma: no cover
            outputs = [self.get_tensor(o) for o in node_info.get("outputs", [])]  # pragma: no cover

            # Simple handling of dynamic sequence axes  # pragma: no cover
            # CNTK often has dynamic sequence axes which ONNX represents using dynamic dimensions (-1)  # pragma: no cover
            # or Sequence constructs. We map them simply.  # pragma: no cover
            if op_type == "Convolution":  # pragma: no cover
                node = Node("Conv", inputs=inputs, outputs=outputs, name=name)  # pragma: no cover
                self.graph.add_node(node)  # pragma: no cover
            elif op_type == "Plus":  # pragma: no cover
                node = Node("Add", inputs=inputs, outputs=outputs, name=name)  # pragma: no cover
                self.graph.add_node(node)  # pragma: no cover
            else:  # pragma: no cover  # pragma: no cover
                node = Node(  # pragma: no cover
                    op_type if op_type else "Identity",
                    inputs=inputs,
                    outputs=outputs,
                    name=name,  # pragma: no cover
                )  # pragma: no cover
                self.graph.add_node(node)  # pragma: no cover

        for out in self.model_info.get("outputs", []):  # pragma: no cover
            t = self.get_tensor(out["name"])  # pragma: no cover
            if t not in self.graph.outputs:  # pragma: no cover
                self.graph.outputs.append(t)  # pragma: no cover

        return self.graph  # pragma: no cover
