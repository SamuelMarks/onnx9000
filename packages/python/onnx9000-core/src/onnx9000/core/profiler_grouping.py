"""Advanced namespace and layer-by-layer grouping for ONNX profiling."""

import json
from typing import Any


def extract_namespace(name: str, delimiter: str = ".") -> list[str]:
    """
    Extracts namespace hierarchy from a node name (e.g. 'model.layer.0.attention.proj').
    Falls back to '/' if '.' isn't used (common in TF/Keras).
    """
    if "." in name:
        return name.split(".")
    if "/" in name:
        return name.split("/")
    return [name]


class HierarchicalProfileNode:
    """Represents the Hierarchical Profile Node class."""

    def __init__(self, name: str):
        """Initializes the instance."""
        self.name = name
        self.macs = 0
        self.flops = 0
        self.params = 0
        self.activation_bytes = 0
        self.children: dict[str, HierarchicalProfileNode] = {}

    def add_stats(self, macs: int, flops: int, params: int, activation_bytes: int):
        """Executes the add stats operation."""
        if isinstance(macs, int):
            self.macs += macs
        if isinstance(flops, int):
            self.flops += flops
        if isinstance(params, int):
            self.params += params
        if isinstance(activation_bytes, int):
            self.activation_bytes += activation_bytes

    def to_dict(self) -> dict[str, Any]:
        """Executes the to dict operation."""
        d = {
            "name": self.name,
            "macs": self.macs,
            "flops": self.flops,
            "params": self.params,
            "activation_bytes": self.activation_bytes,
        }
        if self.children:
            d["children"] = [c.to_dict() for c in self.children.values()]
        return d

    def print_tree(self, indent: int = 0):
        """Executes the print tree operation."""
        pad = "  " * indent
        print(
            f"{pad}- {self.name}: FLOPs={self.flops}, Params={self.params}, Activations={self.activation_bytes}"
        )
        for c in self.children.values():
            c.print_tree(indent + 1)


def group_by_namespace(profiler_result) -> HierarchicalProfileNode:
    """
    Takes a ProfilerResult and groups its flat node_profiles into a hierarchical tree.
    """
    root = HierarchicalProfileNode("root")

    for n in profiler_result.node_profiles:
        name = n.get("name", "unnamed")
        parts = extract_namespace(name)

        current = root
        current.add_stats(
            n.get("macs", 0), n.get("flops", 0), n.get("params", 0), n.get("activation_bytes", 0)
        )

        for p in parts:
            if p not in current.children:
                current.children[p] = HierarchicalProfileNode(p)
            current = current.children[p]
            current.add_stats(
                n.get("macs", 0),
                n.get("flops", 0),
                n.get("params", 0),
                n.get("activation_bytes", 0),
            )

    return root


def export_hierarchical_json(profiler_result, filepath: str):
    """Executes the export hierarchical json operation."""
    tree = group_by_namespace(profiler_result)
    with open(filepath, "w") as f:
        json.dump(tree.to_dict(), f, indent=2)


def to_pandas_dataframe(profiler_result) -> list[dict[str, Any]]:
    """Returns a list of dictionaries ready to be ingested by Pandas DataFrame."""
    df_data = []
    for n in profiler_result.node_profiles:
        namespace = ".".join(extract_namespace(n.get("name", "unnamed"))[:-1])
        df_data.append(
            {
                "Node": n.get("name"),
                "Layer": namespace,
                "OpType": n.get("op_type"),
                "MACs": n.get("macs", 0),
                "FLOPs": n.get("flops", 0),
                "Params": n.get("params", 0),
                "Activations (Bytes)": n.get("activation_bytes", 0),
                "Arithmetic Intensity": n.get("arithmetic_intensity"),
            }
        )
    return df_data


def export_csv(profiler_result, filepath: str):
    """Executes the export csv operation."""
    df_data = to_pandas_dataframe(profiler_result)
    if not df_data:
        return

    keys = df_data[0].keys()
    with open(filepath, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in df_data:
            line = []
            for k in keys:
                line.append(str(row[k]))
            f.write(",".join(line) + "\n")
