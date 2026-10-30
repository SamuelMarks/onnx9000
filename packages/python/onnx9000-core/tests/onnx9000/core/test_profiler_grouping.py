"""Unit tests for hierarchical profile grouping in `onnx9000.core.profiler_grouping`."""

import os
import tempfile

from onnx9000.core.profiler import ProfilerResult
from onnx9000.core.profiler_grouping import (
    HierarchicalProfileNode,
    export_csv,
    export_hierarchical_json,
    extract_namespace,
    group_by_namespace,
    to_pandas_dataframe,
)


def test_profiler_grouping() -> None:
    """Verify HierarchicalProfileNode, extract_namespace, group_by_namespace, export_hierarchical_json, to_pandas_dataframe, and export_csv."""
    assert extract_namespace("a.b.c") == ["a", "b", "c"]
    assert extract_namespace("a/b") == ["a", "b"]

    node = HierarchicalProfileNode("layer0")
    node.add_stats(10, 20, 5, 100)
    assert node.to_dict()["flops"] == 20

    res = ProfilerResult()
    res.node_profiles = [
        {
            "name": "encoder.layer0.conv",
            "op_type": "Conv",
            "macs": 100,
            "flops": 200,
            "params": 50,
            "activation_bytes": 400,
            "arithmetic_intensity": 0.5,
        }
    ]
    tree = group_by_namespace(res)
    assert tree.flops == 200
    assert "encoder" in tree.children

    df = to_pandas_dataframe(res)
    assert len(df) == 1
    assert df[0]["Layer"] == "encoder.layer0"

    with tempfile.TemporaryDirectory() as tmpdir:
        json_p = os.path.join(tmpdir, "prof.json")
        csv_p = os.path.join(tmpdir, "prof.csv")
        export_hierarchical_json(res, json_p)
        export_csv(res, csv_p)
        assert os.path.exists(json_p)
        assert os.path.exists(csv_p)
