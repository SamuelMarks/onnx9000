"""Keras 3 functional model parser."""

from typing import Any

import keras
from onnx9000.converters.tf.parsers import TFGraph, TFNode


class Keras3Parser:
    """Parser for Keras 3 functional models."""

    def __init__(self, model: keras.Model, input_shape: Any = None) -> None:
        """Initialize the Keras 3 parser."""
        self.model = model
        self.input_shape = input_shape
        self.tf_graph = TFGraph()
        self.tensor_to_name: dict[Any, str] = {}
        self._tensor_counter = 0

    def get_tensor_name(self, tensor: Any) -> str:
        """Get or create a unique name for a KerasTensor."""
        if tensor in self.tensor_to_name:
            return self.tensor_to_name[tensor]

        name = getattr(tensor, "name", None)
        if name is None or "keras_tensor" in name:
            name = f"tensor_{self._tensor_counter}"
            self._tensor_counter += 1

        self.tensor_to_name[tensor] = name
        return name

    def parse(self) -> TFGraph:
        """Parse the Keras model into a TFGraph."""
        model = self.model

        # If it's a subclassed model, try to trace it into a functional model
        if not hasattr(model, "operations") or not model.built:
            # We need input shapes to build it.
            # If already built, we might be able to infer them.
            input_shape = self.input_shape or getattr(model, "input_shape", None)
            if input_shape:
                if isinstance(input_shape, list) and isinstance(input_shape[0], tuple):
                    inputs = [keras.Input(batch_shape=s) for s in input_shape]
                else:
                    inputs = keras.Input(batch_shape=input_shape)
                outputs = model(inputs)
                model = keras.Model(inputs, outputs)
            else:
                # Fallback: if we can't trace it, we can't parse it this way.
                raise ValueError(
                    "Subclassed Keras 3 model must be built or have input_shape to be parsed."
                )

        for op in model.operations:
            # Each operation (layer) can have multiple nodes if it's reused.
            for i, node in enumerate(op._inbound_nodes):
                inputs = [self.get_tensor_name(t) for t in node.input_tensors]
                outputs = [self.get_tensor_name(t) for t in node.output_tensors]

                # Extract attributes from layer config
                config = op.get_config()
                attr = {k: v for k, v in config.items() if not k.startswith("_")}

                # Handle weights as additional inputs
                for weight in op.weights:
                    weight_name = f"{op.name}/{weight.name}"
                    weight_node = TFNode(
                        name=weight_name,
                        op="Const",
                        inputs=[],
                        attr={"value": weight.numpy(), "dtype": str(weight.dtype)},
                    )
                    self.tf_graph.nodes.append(weight_node)
                    inputs.append(weight_name)

                # Determine op_type compatible with KERAS_LAYERS_MAPPING
                full_module = op.__class__.__module__
                if full_module.startswith("keras."):
                    module_path = full_module[len("keras.") :]
                else:
                    module_path = full_module

                op_type = f"{module_path}.{op.__class__.__name__}"
                if "InputLayer" in op_type:
                    op_type = "Placeholder"

                node_name = op.name if i == 0 else f"{op.name}_{i}"

                tf_node = TFNode(name=node_name, op=op_type, inputs=inputs, attr=attr)
                self.tf_graph.nodes.append(tf_node)

        return self.tf_graph
