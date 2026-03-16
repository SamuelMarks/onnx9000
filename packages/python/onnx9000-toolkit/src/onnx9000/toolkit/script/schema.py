"""
Registry and validation mechanisms for ONNX operator schemas.
Provides lookup for operation signatures across different opset versions.
"""

import json
from typing import Any, Optional


class OpSchema:
    """Class OpSchema implementation."""

    def __init__(
        self,
        name: str,
        since_version: int,
        inputs: list[str],
        outputs: list[str],
        attributes: list[str],
    ) -> None:
        """Initializes a schema for a specific ONNX operator version."""
        self.name = name
        self.since_version = since_version
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes


class SchemaRegistry:
    """A registry that stores and manages schemas for ONNX operations over different versions."""

    def __init__(self) -> None:
        """Initializes an empty schema registry."""
        self.schemas: dict[str, list[OpSchema]] = {}

    def register(self, schema: OpSchema) -> None:
        """Registers a new operator schema into the registry, maintaining sorted order by version."""
        if schema.name not in self.schemas:
            self.schemas[schema.name] = []
        self.schemas[schema.name].append(schema)
        self.schemas[schema.name].sort(key=lambda s: s.since_version)

    def get_schema(self, op_type: str, version: int) -> Optional[OpSchema]:
        """Retrieves the highest compatible schema for a given operator type and target opset version."""
        if op_type not in self.schemas:
            return None
        candidates = [s for s in self.schemas[op_type] if s.since_version <= version]
        if not candidates:
            return None
        return candidates[-1]

    def load_from_json(self, json_str: str) -> None:
        """Loads and registers operator schemas from a JSON-formatted string."""
        data = json.loads(json_str)
        for item in data:
            self.register(
                OpSchema(
                    item["name"],
                    item.get("since_version", 1),
                    item.get("inputs", []),
                    item.get("outputs", []),
                    item.get("attributes", []),
                )
            )


registry = SchemaRegistry()
registry.register(OpSchema("Add", 14, ["A", "B"], ["C"], []))
registry.register(OpSchema("Relu", 14, ["X"], ["Y"], []))
registry.register(OpSchema("Loop", 13, ["M", "cond", "v_initial"], ["v_final"], ["body"]))
registry.register(OpSchema("If", 13, ["cond"], ["outputs"], ["then_branch", "else_branch"]))
registry.register(OpSchema("Concat", 13, ["inputs"], ["concat_result"], ["axis"]))
_target_opset = 18


def set_target_opset(version: int) -> None:
    """Sets the global target opset version to use for resolving operation schemas."""
    global _target_opset
    _target_opset = version


def get_target_opset() -> int:
    """Retrieves the globally configured target opset version."""
    return _target_opset


def validate_op(op_type: str, inputs: list[Any], attributes: dict[str, Any]) -> None:
    """Validates an operation's instantiation against the globally targeted opset schema."""
    schema = registry.get_schema(op_type, _target_opset)
    if schema is None:
        if op_type in registry.schemas:
            min_version = min((s.since_version for s in registry.schemas[op_type]))
            if _target_opset < min_version:
                raise ValueError(
                    f"Operation '{op_type}' requires opset {min_version}+, but target opset is {_target_opset}"
                )
        return
    for attr in attributes:
        if attr not in schema.attributes:
            import warnings

            warnings.warn(
                f"Attribute '{attr}' is not valid for operation '{op_type}' in opset {_target_opset}.",
                stacklevel=2,
            )
    if op_type == "Squeeze" and _target_opset >= 13 and ("axes" in attributes):
        import warnings

        warnings.warn(
            "Attribute 'axes' will be converted to input for Squeeze in opset 13+", stacklevel=2
        )
