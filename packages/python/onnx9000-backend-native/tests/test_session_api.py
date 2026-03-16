import pytest
from onnx9000.backends.session import InferenceSession, InferenceSessionError, IOBinding
from onnx9000.core.dtypes import DType
from onnx9000.core.execution import ExecutionProvider, RunOptions, SessionOptions
from onnx9000.core.ir import Graph, Node, Tensor


class DummyProvider(ExecutionProvider):
    def get_supported_nodes(self, graph):
        return [n.name or n.op_type for n in graph.nodes if n.op_type == "Identity"]

    def allocate_tensors(self, tensors):
        pass

    def execute(self, graph, context, inputs):
        res = {}
        for node in graph.nodes:
            if node.op_type == "Identity" and all((i in inputs for i in node.inputs)):
                res[node.outputs[0]] = inputs[node.inputs[0]]
        return res


class CPUProvider(ExecutionProvider):
    def get_supported_nodes(self, graph):
        return [n.name or n.op_type for n in graph.nodes if n.op_type in ["Add", "MemcpyToHost"]]

    def allocate_tensors(self, tensors):
        pass

    def execute(self, graph, context, inputs):
        res = {}
        for node in graph.nodes:
            if node.op_type in ["Add", "MemcpyToHost"] and all((i in inputs for i in node.inputs)):
                res[node.outputs[0]] = inputs[node.inputs[0]]
        return res


def test_inference_session_api():
    g = Graph("test")
    g.inputs.append(Tensor("A", (2, 2), DType.FLOAT32))
    g.outputs.append(Tensor("Z", (2, 2), DType.FLOAT32))
    g.add_node(Node("Identity", inputs=["A"], outputs=["Y"], attributes={}))
    g.add_node(Node("Add", inputs=["Y", "A"], outputs=["Z"], attributes={}))
    session_options = SessionOptions()
    session = InferenceSession(
        g, providers=[DummyProvider({}), CPUProvider({})], options=session_options
    )
    assert len(session.graph.nodes) > 2
    assert len(session.get_inputs()) == 1
    assert session.get_inputs()[0].name == "A"
    assert len(session.get_outputs()) == 1
    assert session.get_outputs()[0].name == "Z"
    assert session.get_overridable_initializers() == []
    assert session.get_providers() == ["DummyProvider", "CPUProvider"]
    run_options = RunOptions()
    t = Tensor("A", (2, 2), DType.FLOAT32, data=b"1234")
    res = session.run(["Z"], {"A": t}, run_options=run_options)
    assert len(res) == 1
    io_binding = IOBinding(session)
    io_binding.bind_input("A", "cpu", 0, "float32", (2, 2), 0)
    io_binding.bind_output("Z", "cpu", 0, "float32", (2, 2), 0)
    io_binding.bind_ortvalue_input("A", t)
    io_binding.bind_ortvalue_output("Z", t)
    io_binding.synchronize_inputs()
    io_binding.synchronize_outputs()
    session.run_with_iobinding(io_binding)


def test_session_errors():
    g = Graph("test")
    g.add_node(Node("MissingOp", inputs=[], outputs=["Y"], attributes={}))
    with pytest.raises(InferenceSessionError):
        session = InferenceSession(g, providers=[DummyProvider({})])
