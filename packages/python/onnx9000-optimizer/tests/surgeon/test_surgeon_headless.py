import json
import os
from unittest.mock import MagicMock, patch

from onnx9000.optimizer.surgeon.headless import change_batch, mutate, rename_input


def test_rename_input():
    graph = MagicMock()
    inp = MagicMock()
    inp.name = "old"
    graph.inputs = [inp]
    node = MagicMock()
    node.inputs = ["old", "other"]
    graph.nodes = [node]

    graph = rename_input(graph, "old", "new")

    assert graph.inputs[0].name == "new"
    assert graph.nodes[0].inputs == ["new", "other"]


def test_change_batch():
    graph = MagicMock()
    inp = MagicMock()
    inp.shape = (1, 3, 224, 224)
    graph.inputs = [inp]

    out = MagicMock()
    out.shape = (1, 1000)
    graph.outputs = [out]

    graph = change_batch(graph, 32)
    assert graph.inputs[0].shape == (32, 3, 224, 224)
    assert graph.outputs[0].shape == (32, 1000)


def test_change_batch_string():
    graph = MagicMock()
    inp = MagicMock()
    inp.shape = (1, 3, 224, 224)
    graph.inputs = [inp]

    out = MagicMock()
    out.shape = (1, 1000)
    graph.outputs = [out]

    graph = change_batch(graph, "dynamic_batch")
    assert graph.inputs[0].shape == ("dynamic_batch", 3, 224, 224)
    assert graph.outputs[0].shape == ("dynamic_batch", 1000)


def test_mutate(tmp_path):
    script_path = os.path.join(tmp_path, "mutations.json")
    with open(script_path, "w") as f:
        json.dump([{"action": "remove_node", "node_name": "to_remove"}], f)

    graph = MagicMock()
    node1 = MagicMock()
    node1.name = "keep"
    node2 = MagicMock()
    node2.name = "to_remove"
    graph.nodes = [node1, node2]

    graph = mutate(graph, script_path)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].name == "keep"
