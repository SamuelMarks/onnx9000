"""Sharding abstractions."""

from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.ops import record_op


class PartitionSpec(tuple):
    """Docstring for D101."""

    def __new__(cls, *args):
        """Docstring for D102."""
        return super().__new__(cls, args)


class AutoShardingPass:
    """Docstring for D101."""

    def apply(self, graph: Graph) -> Graph:
        """Docstring for D102."""
        for node in graph.nodes:
            if node.op_type == "MatMul":
                if len(node.inputs) >= 2:
                    sharding_a = getattr(node.inputs[0], "sharding", None)
                    sharding_b = getattr(node.inputs[1], "sharding", None)
                    if sharding_a or sharding_b:
                        if len(node.outputs) > 0:
                            node.outputs[0].sharding = (
                                tuple([None] * len(node.outputs[0].shape))
                                if node.outputs[0].shape
                                else (None,)
                            )
        return graph


class SPMDLoweringPass:
    """Docstring for D101."""

    def apply(self, graph: Graph) -> Graph:
        """Docstring for D102."""
        new_nodes = []
        for node in graph.nodes:
            inserted_before = []
            inserted_after = []

            # FSDP on initializers
            if node.op_type == "MatMul":
                for i, inp in enumerate(node.inputs):
                    if (
                        hasattr(inp, "sharding")
                        and inp.sharding == PartitionSpec("fsdp")
                        and getattr(inp, "is_initializer", False)
                    ):
                        allgather_out = Tensor(
                            name=f"{inp.name}_allgather", shape=inp.shape, dtype=inp.dtype
                        )
                        allgather_node = Node("AllGather", inputs=[inp], outputs=[allgather_out])
                        inserted_before.append(allgather_node)
                        node.inputs[i] = allgather_out
                        discard_node = Node("Discard", inputs=[allgather_out], outputs=[])
                        inserted_after.append(discard_node)

                # TP requirement
                if len(node.inputs) >= 2:
                    weight = node.inputs[1]
                    if hasattr(weight, "sharding"):
                        if weight.sharding == PartitionSpec(None, "tp"):
                            if node.outputs:
                                node.outputs[0].sharding = PartitionSpec(None, "tp")
                        elif weight.sharding == PartitionSpec("tp", None):
                            if node.outputs:
                                orig_out = node.outputs[0]
                                matmul_out = Tensor(
                                    name=f"{orig_out.name}_partial",
                                    shape=orig_out.shape,
                                    dtype=orig_out.dtype,
                                )
                                node.outputs[0] = matmul_out
                                allreduce_node = Node(
                                    "AllReduce",
                                    inputs=[matmul_out],
                                    outputs=[orig_out],
                                    attributes={"op": Attribute(name="op", value="sum")},
                                )
                                inserted_after.append(allreduce_node)

            # EP requirement
            has_ep = False
            for inp in node.inputs:
                if hasattr(inp, "sharding") and inp.sharding == PartitionSpec("ep", None, None):
                    has_ep = True
                    break
            if has_ep:
                new_inputs = []
                for inp in node.inputs:
                    a2a_out = Tensor(name=f"{inp.name}_a2a_in", shape=inp.shape, dtype=inp.dtype)
                    a2a_in_node = Node("AllToAll", inputs=[inp], outputs=[a2a_out])
                    inserted_before.append(a2a_in_node)
                    new_inputs.append(a2a_out)
                node.inputs = new_inputs

                if node.outputs:
                    orig_out = node.outputs[0]
                    comp_out = Tensor(
                        name=f"{orig_out.name}_comp", shape=orig_out.shape, dtype=orig_out.dtype
                    )
                    node.outputs[0] = comp_out
                    a2a_out_node = Node("AllToAll", inputs=[comp_out], outputs=[orig_out])
                    inserted_after.append(a2a_out_node)

            # CP requirement
            if node.op_type == "FlashAttention":
                has_cp = False
                for inp in node.inputs:
                    if hasattr(inp, "sharding") and inp.sharding == PartitionSpec(None, "cp", None):
                        has_cp = True
                        break
                if has_cp:
                    recv_out = Tensor(name=f"recv_cp_{node.name}")
                    recv_node = Node("Recv", inputs=[], outputs=[recv_out])
                    inserted_before.append(recv_node)

                    send_node = Node(
                        "Send", inputs=[node.outputs[0] if node.outputs else recv_out], outputs=[]
                    )
                    inserted_after.append(send_node)

            # PP requirement
            if (
                node.outputs
                and hasattr(node.outputs[0], "sharding")
                and node.outputs[0].sharding == PartitionSpec("pp")
            ):
                orig_out = node.outputs[0]
                node.outputs[0] = Tensor(
                    name=f"{orig_out.name}_stage", shape=orig_out.shape, dtype=orig_out.dtype
                )
                node.outputs[0].sharding = PartitionSpec("pp")

                send_node = Node("Send", inputs=[node.outputs[0]], outputs=[])
                inserted_after.append(send_node)

                recv_node = Node("Recv", inputs=[], outputs=[orig_out])
                inserted_after.append(recv_node)

            new_nodes.extend(inserted_before)
            new_nodes.append(node)
            new_nodes.extend(inserted_after)

        graph.nodes = new_nodes
        return graph


def all_reduce(x: Tensor, group: str = "world") -> Tensor:
    """Docstring for D103."""
    return record_op("AllReduce", [x], {"group": group})


def all_gather(x: Tensor, group: str = "world") -> Tensor:
    """Docstring for D103."""
    return record_op("AllGather", [x], {"group": group})


def reduce_scatter(x: Tensor, group: str = "world") -> Tensor:
    """Docstring for D103."""
    return record_op("ReduceScatter", [x], {"group": group})


def all_to_all(x: Tensor, group: str = "world") -> Tensor:
    """Docstring for D103."""
    return record_op("AllToAll", [x], {"group": group})
