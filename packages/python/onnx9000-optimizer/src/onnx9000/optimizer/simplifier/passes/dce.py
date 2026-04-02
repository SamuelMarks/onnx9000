"""Eliminate dead code from the ONNX computational graph."""

from __future__ import annotations

"""Provides dce.py module functionality."""

import logging

from onnx9000.core.ir import Graph
from onnx9000.optimizer.simplifier.passes.base import GraphPass

logger = logging.getLogger(__name__)


class DCEPass(GraphPass):
    """Dead Code Elimination (DCE).

    Removes nodes whose outputs are never consumed by any other node
    and are not in the graph's explicitly defined outputs.
    """

    def __init__(
        self,
        unused_inputs_to_prune: list[str] | None = None,
        nodes_to_preserve: list[str] | None = None,
    ):
        """Initialize dead code elimination pass."""
        self.nodes_to_preserve = set(nodes_to_preserve or [])
        super().__init__()
        self.unused_inputs_to_prune = set(unused_inputs_to_prune or [])

    def run(self, graph: Graph) -> bool:
        """Implement the run method or operation."""
        changed = False
        while True:
            local_changed = self._run_once(graph)

            # Recurse into subgraphs
            for node in graph.nodes:
                for attr_name, attr in node.attributes.items():
                    if hasattr(attr, "attr_type") and attr.attr_type == "GRAPH":
                        sub_changed = self.run(attr.value)
                        if sub_changed:
                            local_changed = True

            if not local_changed:
                break
            changed = True
        return changed

    def _run_once(self, graph: Graph) -> bool:
        """Implement the _run_once method or operation."""
        changed = False

        # Identify nodes that produce each tensor
        producers = {}
        for node in graph.nodes:
            for out in node.outputs:
                producers[out] = node

        # Compute reachability from outputs backwards
        reachable_tensors = set()
        for out in graph.outputs:
            reachable_tensors.add(getattr(out, "name", out) if not isinstance(out, str) else out)

        for node in graph.nodes:
            if node.name in self.nodes_to_preserve:
                for out in node.outputs:
                    reachable_tensors.add(
                        getattr(out, "name", out) if not isinstance(out, str) else out
                    )

        stack = list(reachable_tensors)
        while stack:
            t_name = stack.pop()
            producer = producers.get(t_name)
            if producer:
                for inp in producer.inputs:
                    if inp not in reachable_tensors:
                        reachable_tensors.add(inp)
                        stack.append(inp)

        new_nodes = []
        for node in graph.nodes:
            # We keep a node if ANY of its outputs are reachable
            if any(out in reachable_tensors for out in node.outputs):
                new_nodes.append(node)
            else:
                logger.info(f"Eliminated dead node {node.name} ({node.op_type})")
                changed = True

        graph.nodes = new_nodes

        consumed_initializers = {init for init in graph.initializers if init in reachable_tensors}
        if len(consumed_initializers) < len(graph.initializers):
            graph.initializers = list(consumed_initializers)
            changed = True

        new_tensors = {k: v for k, v in graph.tensors.items() if k in reachable_tensors}
        if len(new_tensors) < len(graph.tensors):
            graph.tensors = new_tensors
            changed = True

        new_value_info = [
            vi
            for vi in getattr(graph, "value_info", [])
            if getattr(vi, "name", vi) in reachable_tensors
        ]
        if len(new_value_info) < len(getattr(graph, "value_info", [])):
            graph.value_info = new_value_info
            changed = True

        if self.unused_inputs_to_prune:
            new_inputs = [
                inp
                for inp in graph.inputs
                if getattr(inp, "name", inp) in reachable_tensors
                or getattr(inp, "name", inp) not in self.unused_inputs_to_prune
            ]
            if len(new_inputs) < len(graph.inputs):
                logger.info("Pruned unused explicitly specified graph inputs.")
                graph.inputs = new_inputs
                changed = True

        return changed


class IdentityEliminationPass(GraphPass):
    """Detects and removes explicit Identity nodes and redundant operations.

    like Cast(Cast(X)), Reshape(Reshape(X)), Transpose(Transpose(X)), etc.
    Rewires inputs to consumers.
    """

    def run(self, graph: Graph) -> bool:
        """Implement the run method or operation."""
        changed = False
        while True:
            local_changed = self._run_once(graph)

            # Recurse into subgraphs
            for node in graph.nodes:
                for attr_name, attr in node.attributes.items():
                    if hasattr(attr, "attr_type") and attr.attr_type == "GRAPH":
                        sub_changed = self.run(attr.value)
                        if sub_changed:
                            local_changed = True

            if not local_changed:
                break
            changed = True
        return changed

    def _run_once(self, graph: Graph) -> bool:
        """Implement the _run_once method or operation."""
        changed = False
        producers = {}
        for node in graph.nodes:
            for out in node.outputs:
                producers[out] = node
        new_nodes = []
        for node in graph.nodes:
            eliminated = False
            if node.op_type == "Dropout":
                self._rewire(graph, node.outputs[0], node.inputs[0])
                if len(node.outputs) > 1 and node.outputs[1]:
                    logger.debug("Handled via replacement implicitly.")
                eliminated = True
                changed = True
                logger.info(f"Eliminated Dropout node {node.name}")
            elif node.op_type == "Identity":
                self._rewire(graph, node.outputs[0], node.inputs[0])
                eliminated = True
                changed = True
                logger.info(f"Eliminated Identity node {node.name}")
            elif node.op_type == "Concat" and len(node.inputs) == 1:
                self._rewire(graph, node.outputs[0], node.inputs[0])
                eliminated = True
                changed = True
                logger.info(f"Eliminated single-input Concat node {node.name}")
            elif node.op_type in ("Max", "Min", "And", "Or"):
                if len(node.inputs) == 2 and node.inputs[0] == node.inputs[1]:
                    self._rewire(graph, node.outputs[0], node.inputs[0])
                    eliminated = True
                    changed = True
                    logger.info(f"Eliminated double {node.op_type} at {node.name}")
            elif node.op_type == "Where":
                if len(node.inputs) == 3 and node.inputs[1] == node.inputs[2]:
                    self._rewire(graph, node.outputs[0], node.inputs[1])
                    eliminated = True
                    changed = True
                    logger.info(f"Eliminated Where(C, X, X) at {node.name}")
            elif node.op_type in ("Neg", "Not"):
                producer = producers.get(node.inputs[0])
                if producer and producer.op_type == node.op_type:
                    self._rewire(graph, node.outputs[0], producer.inputs[0])
                    eliminated = True
                    changed = True
                    logger.info(f"Eliminated double {node.op_type} at {node.name}")

            elif node.op_type in ("Add", "Mul"):
                if len(node.inputs) == 2:
                    c1_name = None
                    x_name = None
                    for i in range(2):
                        t = graph.tensors.get(node.inputs[i])
                        if t and getattr(t, "is_initializer", False):
                            c1_name = node.inputs[i]
                            x_name = node.inputs[1 - i]
                            break
                    if c1_name and x_name:
                        producer = producers.get(x_name)
                        if producer and producer.op_type == node.op_type:
                            c2_name = None
                            inner_x = None
                            for i in range(2):
                                t2_cand = graph.tensors.get(producer.inputs[i])
                                if t2_cand and getattr(t2_cand, "is_initializer", False):
                                    c2_name = producer.inputs[i]
                                    inner_x = producer.inputs[1 - i]
                                    break
                            if c2_name and inner_x:
                                import numpy as np

                                t2_data = graph.tensors[c1_name].data
                                t1_data = graph.tensors[c2_name].data
                                if isinstance(t1_data, np.ndarray) and isinstance(
                                    t2_data, np.ndarray
                                ):
                                    if node.op_type == "Add":
                                        new_data = t1_data + t2_data
                                    else:
                                        new_data = t1_data * t2_data

                                    new_c_name = f"{node.name}_folded_c"
                                    from onnx9000.core.ir import Constant

                                    new_c = Constant(
                                        new_c_name,
                                        values=new_data,
                                        shape=new_data.shape,
                                        dtype=graph.tensors[c1_name].dtype,
                                    )
                                    graph.add_tensor(new_c)
                                    graph.initializers.append(new_c_name)

                                    node.inputs = [inner_x, new_c_name]
                                    changed = True
                                    logger.info(f"Folded sequential {node.op_type} at {node.name}")
                        elif producer and producer.op_type == "Add" and node.op_type == "Mul":
                            # Mul(Add(X, C1_inner), C1_outer) -> Add(Mul(X, C1_outer), C1_inner*C1_outer)
                            c1_name_inner = None
                            inner_x = None
                            for i in range(2):
                                t1_cand = graph.tensors.get(producer.inputs[i])
                                if t1_cand and getattr(t1_cand, "is_initializer", False):
                                    c1_name_inner = producer.inputs[i]
                                    inner_x = producer.inputs[1 - i]
                                    break
                            if c1_name_inner and inner_x:
                                import numpy as np

                                t2_data = graph.tensors[c1_name].data
                                t1_data = graph.tensors[c1_name_inner].data
                                if isinstance(t1_data, np.ndarray) and isinstance(
                                    t2_data, np.ndarray
                                ):
                                    new_c1_data = t1_data * t2_data
                                    new_c1_name = f"{node.name}_dist_c1"
                                    from onnx9000.core.ir import Constant, Node

                                    new_c1 = Constant(
                                        new_c1_name,
                                        values=new_c1_data,
                                        shape=new_c1_data.shape,
                                        dtype=graph.tensors[c1_name].dtype,
                                    )
                                    graph.add_tensor(new_c1)
                                    graph.initializers.append(new_c1_name)

                                    mul_out_name = f"{node.name}_dist_mul_out"
                                    new_mul = Node(
                                        op_type="Mul",
                                        inputs=[inner_x, c1_name],
                                        outputs=[mul_out_name],
                                        name=f"{node.name}_dist_mul",
                                    )
                                    graph.add_node(new_mul)

                                    node.op_type = "Add"
                                    node.inputs = [mul_out_name, new_c1_name]
                                    changed = True
                                    logger.info(f"Distributed Mul(Add(X, C1), C2) at {node.name}")
            elif node.op_type == "Abs":
                producer = producers.get(node.inputs[0])
                if producer and producer.op_type == "Abs":
                    node.inputs[0] = producer.inputs[0]
                    changed = True
                    logger.info(f"Simplified chained Abs at {node.name}")
            elif node.op_type == "Exp":
                producer = producers.get(node.inputs[0])
                if producer and producer.op_type == "Log":
                    self._rewire(graph, node.outputs[0], producer.inputs[0])
                    eliminated = True
                    changed = True
                    logger.info(f"Eliminated Exp(Log(X)) at {node.name}")
            elif node.op_type == "Log":
                producer = producers.get(node.inputs[0])
                if producer and producer.op_type == "Exp":
                    self._rewire(graph, node.outputs[0], producer.inputs[0])
                    eliminated = True
                    changed = True
                    logger.info(f"Eliminated Log(Exp(X)) at {node.name}")
            elif node.op_type == "Sqrt":
                producer = producers.get(node.inputs[0])
                if producer and producer.op_type == "Pow":
                    if len(producer.inputs) > 1:
                        node.op_type = "Abs"
                        node.inputs = [producer.inputs[0]]
                        changed = True
                        logger.info(f"Simplified Sqrt(Pow(X, 2)) to Abs(X) at {node.name}")
                        logger.debug("Handled via replacement implicitly.")
            elif node.op_type in ("Cast", "CastLike"):
                producer = producers.get(node.inputs[0])
                if producer and producer.op_type in ("Cast", "CastLike"):
                    node.inputs[0] = producer.inputs[0]
                    changed = True
                    logger.info(f"Simplified chained Cast at {node.name}")
            elif node.op_type == "Reshape":
                if len(node.inputs) > 1:
                    shape_producer = producers.get(node.inputs[1])
                    if (
                        shape_producer
                        and shape_producer.op_type == "Shape"
                        and shape_producer.inputs[0] == node.inputs[0]
                    ):
                        self._rewire(graph, node.outputs[0], node.inputs[0])
                        eliminated = True
                        changed = True
                        logger.info(f"Eliminated redundant Reshape(X, Shape(X)) at {node.name}")
                        continue

                producer = producers.get(node.inputs[0])
                if producer and producer.op_type == "Reshape":
                    node.inputs[0] = producer.inputs[0]
                    changed = True
                    logger.info(f"Simplified chained Reshape at {node.name}")
            elif node.op_type == "Expand":
                if len(node.inputs) > 1:
                    shape_producer = producers.get(node.inputs[1])
                    if (
                        shape_producer
                        and shape_producer.op_type == "Shape"
                        and shape_producer.inputs[0] == node.inputs[0]
                    ):
                        self._rewire(graph, node.outputs[0], node.inputs[0])
                        eliminated = True
                        changed = True
                        logger.info(f"Eliminated redundant Expand(X, Shape(X)) at {node.name}")
                if not eliminated:
                    producer = producers.get(node.inputs[0])
                    if producer and producer.op_type == "Expand":
                        # Chained expand: Expand(Expand(X, shape1), shape2) -> Expand(X, shape2)
                        node.inputs[0] = producer.inputs[0]
                        changed = True
                        logger.info(f"Simplified chained Expand at {node.name}")
            elif node.op_type == "Pad":
                if len(node.inputs) > 1:
                    pads_tensor = graph.tensors.get(node.inputs[1])
                    if pads_tensor and pads_tensor.is_initializer:
                        import numpy as np

                        if np.all(pads_tensor.data == 0):
                            self._rewire(graph, node.outputs[0], node.inputs[0])
                            eliminated = True
                            changed = True
                            logger.info(f"Eliminated zero Pad at {node.name}")
            elif node.op_type == "Tile":
                if len(node.inputs) > 1:
                    repeats_tensor = graph.tensors.get(node.inputs[1])
                    if repeats_tensor and repeats_tensor.is_initializer:
                        import numpy as np

                        if np.all(repeats_tensor.data == 1):
                            self._rewire(graph, node.outputs[0], node.inputs[0])
                            eliminated = True
                            changed = True
                            logger.info(f"Eliminated identity Tile at {node.name}")
            elif node.op_type == "Slice":
                if len(node.inputs) > 2:
                    starts_tensor = graph.tensors.get(node.inputs[1])
                    ends_tensor = graph.tensors.get(node.inputs[2])
                    if (
                        starts_tensor
                        and ends_tensor
                        and starts_tensor.is_initializer
                        and ends_tensor.is_initializer
                    ):
                        import numpy as np

                        if np.all(starts_tensor.data == 0) and np.all(
                            ends_tensor.data >= 2147483647
                        ):
                            # It's a full slice on all provided axes. Note: could check steps as well.
                            steps_ok = True
                            if len(node.inputs) > 4:
                                steps_tensor = graph.tensors.get(node.inputs[4])
                                if steps_tensor and steps_tensor.is_initializer:
                                    if not np.all(steps_tensor.data == 1):
                                        steps_ok = False
                                else:
                                    if node.inputs[4] != "":
                                        steps_ok = False  # Dynamic steps, can't eliminate
                            if steps_ok:
                                self._rewire(graph, node.outputs[0], node.inputs[0])
                                eliminated = True
                                changed = True
                                logger.info(f"Eliminated full Slice at {node.name}")
            elif node.op_type in ("ReduceSum", "ReduceMean"):
                inp = node.inputs[0]
                is_scalar = False
                if inp in graph.tensors and getattr(graph.tensors[inp], "shape", None) == ():
                    is_scalar = True
                else:
                    for vi in graph.inputs + graph.outputs:
                        if getattr(vi, "name", None) == inp and getattr(vi, "shape", None) == ():
                            is_scalar = True
                            break
                if is_scalar:
                    self._rewire(graph, node.outputs[0], node.inputs[0])
                    eliminated = True
                    changed = True
                    logger.info(f"Eliminated {node.op_type} on scalar at {node.name}")
            elif node.op_type == "Transpose":
                perm = node.attributes.get("perm")
                if perm is not None and list(perm) == list(range(len(perm))):
                    self._rewire(graph, node.outputs[0], node.inputs[0])
                    eliminated = True
                    changed = True
                    logger.info(f"Eliminated Identity Transpose node {node.name}")
                    continue
                producer = producers.get(node.inputs[0])
                if producer and producer.op_type == "Transpose":
                    perm1 = producer.attributes.get("perm")
                    perm2 = node.attributes.get("perm")
                    if perm1 is not None and perm2 is not None:
                        fused_perm = [perm1[p] for p in perm2]
                        if fused_perm == list(range(len(fused_perm))):
                            self._rewire(graph, node.outputs[0], producer.inputs[0])
                            eliminated = True
                        else:
                            node.attributes["perm"] = fused_perm
                            node.inputs[0] = producer.inputs[0]
                        changed = True
                        logger.info(f"Simplified chained Transpose at {node.name}")
            elif node.op_type == "Unsqueeze":
                producer = producers.get(node.inputs[0])
                if producer and producer.op_type == "Squeeze":
                    axes1 = producer.attributes.get("axes")
                    axes2 = node.attributes.get("axes")
                    if axes1 is not None and axes2 is not None and (list(axes1) == list(axes2)):
                        self._rewire(graph, node.outputs[0], producer.inputs[0])
                        eliminated = True
                        changed = True
                        logger.info(f"Eliminated Squeeze-Unsqueeze pair at {node.name}")
            if not eliminated:
                new_nodes.append(node)
        graph.nodes = new_nodes
        return changed

    def _rewire(self, graph: Graph, old_name: str, new_name: str) -> None:
        """Implement the _rewire method or operation."""
        for node in graph.nodes:
            for i, inp in enumerate(node.inputs):
                if inp == old_name:
                    node.inputs[i] = new_name
        for i, out in enumerate(graph.outputs):
            if out == old_name:
                graph.outputs[i] = new_name


def dead_code_elimination(
    graph: Graph,
    unused_inputs_to_prune: list[str] | None = None,
    nodes_to_preserve: list[str] | None = None,
) -> None:
    """Implement the dead_code_elimination method or operation."""
    while True:
        dce_changed = DCEPass(unused_inputs_to_prune, nodes_to_preserve).run(graph)
        id_changed = IdentityEliminationPass().run(graph)
        cf_changed = ControlFlowFoldingPass().run(graph)
        if not dce_changed and not id_changed and not cf_changed:
            break


class ControlFlowFoldingPass(GraphPass):
    """Folds 'If' and 'Loop' control flow nodes explicitly when their conditions are statically known constants."""

    def run(self, graph: Graph) -> bool:
        """Implement the run method or operation."""
        changed = False
        while True:
            local_changed = self._run_once(graph)

            # Recurse into subgraphs
            for node in graph.nodes:
                for attr_name, attr in node.attributes.items():
                    if hasattr(attr, "attr_type") and attr.attr_type == "GRAPH":
                        sub_changed = self.run(attr.value)
                        if sub_changed:
                            local_changed = True

            if not local_changed:
                break
            changed = True
        return changed

    def _run_once(self, graph: Graph) -> bool:
        """Implement the _run_once method or operation."""
        changed = False
        nodes_to_remove = set()

        for node in graph.nodes:
            if node.op_type == "If":
                cond_name = node.inputs[0]
                cond_tensor = graph.tensors.get(cond_name)
                if cond_tensor and cond_tensor.is_initializer:
                    import numpy as np

                    cond_val = cond_tensor.data
                    if isinstance(cond_val, np.ndarray):
                        cond_bool = bool(cond_val.item() if cond_val.size == 1 else cond_val[0])
                    else:
                        cond_bool = bool(cond_val)

                    branch_attr = "then_branch" if cond_bool else "else_branch"
                    subgraph = node.attributes[branch_attr].value

                    prefix = f"{node.name}_fold_"
                    local_tensors = set(subgraph.initializers)
                    for sub_node in subgraph.nodes:
                        for out in sub_node.outputs:
                            local_tensors.add(out)

                    tensor_map = {name: f"{prefix}{name}" for name in local_tensors}

                    # Add initializers and tensors from subgraph
                    for init_name in subgraph.initializers:
                        new_name = tensor_map[init_name]
                        t = subgraph.tensors[init_name].copy()
                        t.name = new_name
                        graph.add_tensor(t)
                        graph.initializers.append(new_name)

                    for t_name, t in subgraph.tensors.items():
                        if t_name in tensor_map and t_name not in subgraph.initializers:
                            new_name = tensor_map[t_name]
                            new_t = t.copy()
                            new_t.name = new_name
                            graph.add_tensor(new_t)

                    # Add nodes
                    from onnx9000.core.ir import Attribute, Node

                    for sub_node in subgraph.nodes:
                        new_inputs = [tensor_map.get(inp, inp) for inp in sub_node.inputs]
                        new_outputs = [tensor_map.get(out, out) for out in sub_node.outputs]
                        new_attrs = {
                            k: Attribute(v.name, v.attr_type, v.value)
                            for k, v in sub_node.attributes.items()
                        }
                        new_node = Node(
                            op_type=sub_node.op_type,
                            inputs=new_inputs,
                            outputs=new_outputs,
                            attributes=new_attrs,
                            name=f"{prefix}{sub_node.name}",
                            domain=sub_node.domain,
                        )
                        graph.add_node(new_node)

                    # Rewire If node outputs to the subgraph outputs
                    for i, out_name in enumerate(node.outputs):
                        sub_out_name = getattr(subgraph.outputs[i], "name", subgraph.outputs[i])
                        mapped_sub_out = tensor_map.get(sub_out_name, sub_out_name)
                        self._rewire(graph, out_name, mapped_sub_out)

                    nodes_to_remove.add(node)
                    changed = True
                    logger.info(f"Folded If node {node.name} (Condition: {cond_bool})")
            elif node.op_type == "Loop":
                max_trip_count_name = node.inputs[0]
                cond_name = node.inputs[1] if len(node.inputs) > 1 else None

                trip_val = None
                if max_trip_count_name:
                    trip_tensor = graph.tensors.get(max_trip_count_name)
                    if trip_tensor and trip_tensor.is_initializer:
                        import numpy as np

                        trip_val = trip_tensor.data
                        if isinstance(trip_val, np.ndarray):
                            trip_val = int(trip_val.item() if trip_val.size == 1 else trip_val[0])
                        else:
                            trip_val = int(trip_val)

                cond_val = None
                if cond_name:
                    cond_tensor = graph.tensors.get(cond_name)
                    if cond_tensor and cond_tensor.is_initializer:
                        import numpy as np

                        cond_data = cond_tensor.data
                        if isinstance(cond_data, np.ndarray):
                            cond_val = bool(
                                cond_data.item() if cond_data.size == 1 else cond_data[0]
                            )
                        else:
                            cond_val = bool(cond_data)

                if trip_val == 0 or cond_val is False:
                    # route initial states to outputs
                    v_initials = node.inputs[2:]
                    v_finals = node.outputs[: len(v_initials)]
                    for i, v_init in enumerate(v_initials):
                        self._rewire(graph, v_finals[i], v_init)

                    # create SequenceEmpty for scan_outputs
                    from onnx9000.core.ir import Node

                    scan_outputs = node.outputs[len(v_initials) :]
                    for i, scan_out in enumerate(scan_outputs):
                        seq_node = Node(
                            op_type="SequenceEmpty",
                            inputs=[],
                            outputs=[scan_out],
                            name=f"{node.name}_seq_{i}",
                        )
                        graph.add_node(seq_node)

                    nodes_to_remove.add(node)
                    changed = True
                    logger.info(f"Folded Loop node {node.name} (max_trip_count=0)")
                elif (
                    trip_val is not None
                    and trip_val < 10
                    and len(node.attributes["body"].value.nodes) * trip_val < 1000
                ):
                    # Unroll loop natively
                    subgraph = node.attributes["body"].value
                    prefix = f"{node.name}_unroll_"

                    body_inputs = [getattr(inp, "name", inp) for inp in subgraph.inputs]
                    body_outputs = [getattr(out, "name", out) for out in subgraph.outputs]

                    iter_num_name = body_inputs[0]
                    cond_in_name = body_inputs[1]
                    v_in_names = body_inputs[2:]

                    # Initialize states
                    current_v = list(node.inputs[2:])
                    current_cond = node.inputs[1] if len(node.inputs) > 1 and node.inputs[1] else ""

                    scan_outputs_accum = [[] for _ in range(len(node.outputs) - len(v_in_names))]

                    import numpy as np
                    from onnx9000.core.ir import Attribute, Constant, Node

                    for step in range(trip_val):
                        step_prefix = f"{prefix}{step}_"
                        tensor_map = {}

                        iter_tensor_name = f"{step_prefix}iter_num"
                        iter_tensor = Constant(
                            iter_tensor_name,
                            values=np.array([step], dtype=np.int64),
                            shape=(1,),
                            dtype="int64",
                        )
                        graph.add_tensor(iter_tensor)
                        graph.initializers.append(iter_tensor_name)
                        tensor_map[iter_num_name] = iter_tensor_name

                        if current_cond:
                            tensor_map[cond_in_name] = current_cond

                        for i, v_in in enumerate(v_in_names):
                            tensor_map[v_in] = current_v[i]

                        local_tensors = set(subgraph.initializers)
                        for sub_node in subgraph.nodes:
                            for out in sub_node.outputs:
                                local_tensors.add(out)

                        for name in local_tensors:
                            tensor_map[name] = f"{step_prefix}{name}"

                        for init_name in subgraph.initializers:
                            new_name = tensor_map[init_name]
                            t = subgraph.tensors[init_name].copy()
                            t.name = new_name
                            graph.add_tensor(t)
                            graph.initializers.append(new_name)

                        for sub_node in subgraph.nodes:
                            new_inputs = [tensor_map.get(inp, inp) for inp in sub_node.inputs]
                            new_outputs = [tensor_map.get(out, out) for out in sub_node.outputs]
                            new_attrs = {
                                k: Attribute(v.name, v.attr_type, v.value)
                                for k, v in sub_node.attributes.items()
                            }
                            new_node = Node(
                                op_type=sub_node.op_type,
                                inputs=new_inputs,
                                outputs=new_outputs,
                                attributes=new_attrs,
                                name=f"{step_prefix}{sub_node.name}",
                                domain=sub_node.domain,
                            )
                            graph.add_node(new_node)

                        body_v_outs = body_outputs[1 : 1 + len(v_in_names)]
                        body_scan_outs = body_outputs[1 + len(v_in_names) :]

                        # Next iteration inputs
                        current_v = [tensor_map.get(o, o) for o in body_v_outs]
                        current_cond = tensor_map.get(body_outputs[0], body_outputs[0])

                        # Accumulate scan outputs
                        for i, scan_out in enumerate(body_scan_outs):
                            scan_outputs_accum[i].append(tensor_map.get(scan_out, scan_out))

                    # Final assignments
                    for i, v_final in enumerate(node.outputs[: len(v_in_names)]):
                        self._rewire(graph, v_final, current_v[i])

                    # Reconstruct sequences using SequenceConstruct or Concat if we unrolled completely
                    # Actually for ONNX loop, scan outputs are tensors with an extra dimension
                    for i, scan_final in enumerate(node.outputs[len(v_in_names) :]):
                        if len(scan_outputs_accum[i]) == 1:
                            # Just an unsqueeze
                            from onnx9000.core.ir import Node

                            axes_name = f"{prefix}axes_{i}"
                            axes_tensor = Constant(
                                axes_name,
                                values=np.array([0], dtype=np.int64),
                                shape=(1,),
                                dtype="int64",
                            )
                            graph.add_tensor(axes_tensor)
                            graph.initializers.append(axes_name)

                            unsqueeze_node = Node(
                                op_type="Unsqueeze",
                                inputs=[scan_outputs_accum[i][0], axes_name],
                                outputs=[scan_final],
                                name=f"{prefix}unsqueeze_{i}",
                            )
                            graph.add_node(unsqueeze_node)
                        elif len(scan_outputs_accum[i]) > 1:
                            # First we sequence construct then concat or sequence insert?
                            # Easiest: Unsqueeze each, then Concat
                            concat_inputs = []
                            for step, item in enumerate(scan_outputs_accum[i]):
                                axes_name = f"{prefix}axes_{i}_{step}"
                                axes_tensor = Constant(
                                    axes_name,
                                    values=np.array([0], dtype=np.int64),
                                    shape=(1,),
                                    dtype="int64",
                                )
                                graph.add_tensor(axes_tensor)
                                graph.initializers.append(axes_name)

                                unsq_out = f"{prefix}unsq_out_{i}_{step}"
                                unsqueeze_node = Node(
                                    op_type="Unsqueeze",
                                    inputs=[item, axes_name],
                                    outputs=[unsq_out],
                                    name=f"{prefix}unsqueeze_{i}_{step}",
                                )
                                graph.add_node(unsqueeze_node)
                                concat_inputs.append(unsq_out)

                            concat_node = Node(
                                op_type="Concat",
                                inputs=concat_inputs,
                                outputs=[scan_final],
                                attributes={"axis": Attribute("axis", "INT", 0)},
                                name=f"{prefix}concat_{i}",
                            )
                            graph.add_node(concat_node)

                    nodes_to_remove.add(node)
                    changed = True
                    logger.info(
                        f"Unrolled Loop node {node.name} natively (max_trip_count={trip_val})"
                    )
                    # inject loop body directly into parent graph
                    subgraph = node.attributes["body"].value
                    prefix = f"{node.name}_fold_loop_"

                    body_inputs = [getattr(inp, "name", inp) for inp in subgraph.inputs]
                    body_outputs = [getattr(out, "name", out) for out in subgraph.outputs]

                    iter_num_name = body_inputs[0]
                    cond_in_name = body_inputs[1]
                    v_in_names = body_inputs[2:]

                    tensor_map = {}

                    from onnx9000.core.ir import Attribute, Constant, Node

                    iter_tensor_name = f"{prefix}iter_num"
                    iter_tensor = Constant(
                        iter_tensor_name,
                        values=np.array([0], dtype=np.int64),
                        shape=(1,),
                        dtype="int64",
                    )
                    graph.add_tensor(iter_tensor)
                    graph.initializers.append(iter_tensor_name)
                    tensor_map[iter_num_name] = iter_tensor_name

                    tensor_map[cond_in_name] = node.inputs[1] if node.inputs[1] else ""
                    for i, v_in in enumerate(v_in_names):
                        tensor_map[v_in] = node.inputs[i + 2]

                    local_tensors = set(subgraph.initializers)
                    for sub_node in subgraph.nodes:
                        for out in sub_node.outputs:
                            local_tensors.add(out)

                    for name in local_tensors:
                        tensor_map[name] = f"{prefix}{name}"

                    # Add sub-graph initializers
                    for init_name in subgraph.initializers:
                        new_name = tensor_map[init_name]
                        t = subgraph.tensors[init_name].copy()
                        t.name = new_name
                        graph.add_tensor(t)
                        graph.initializers.append(new_name)

                    # Add nodes
                    for sub_node in subgraph.nodes:
                        new_inputs = [tensor_map.get(inp, inp) for inp in sub_node.inputs]
                        new_outputs = [tensor_map.get(out, out) for out in sub_node.outputs]
                        new_attrs = {
                            k: Attribute(v.name, v.attr_type, v.value)
                            for k, v in sub_node.attributes.items()
                        }
                        new_node = Node(
                            op_type=sub_node.op_type,
                            inputs=new_inputs,
                            outputs=new_outputs,
                            attributes=new_attrs,
                            name=f"{prefix}{sub_node.name}",
                            domain=sub_node.domain,
                        )
                        graph.add_node(new_node)

                    body_v_outs = body_outputs[1 : 1 + len(v_in_names)]
                    body_scan_outs = body_outputs[1 + len(v_in_names) :]

                    for i, v_final in enumerate(node.outputs[: len(body_v_outs)]):
                        sub_out_name = body_v_outs[i]
                        self._rewire(graph, v_final, tensor_map.get(sub_out_name, sub_out_name))

                    for i, scan_final in enumerate(node.outputs[len(body_v_outs) :]):
                        sub_scan_name = body_scan_outs[i]
                        mapped_sub_scan = tensor_map.get(sub_scan_name, sub_scan_name)
                        seq_node = Node(
                            op_type="SequenceConstruct",
                            inputs=[mapped_sub_scan],
                            outputs=[scan_final],
                            name=f"{node.name}_seq_{i}",
                        )
                        graph.add_node(seq_node)

                    nodes_to_remove.add(node)
                    changed = True
                    logger.info(f"Folded Loop node {node.name} (max_trip_count=1)")

        if changed:
            graph.nodes = [n for n in graph.nodes if n not in nodes_to_remove]

        return changed

    def _rewire(self, graph: Graph, old_name: str, new_name: str) -> None:
        """Implement the _rewire method or operation."""
        for node in graph.nodes:
            for i, inp in enumerate(node.inputs):
                if inp == old_name:
                    node.inputs[i] = new_name
        for i, out in enumerate(graph.outputs):
            if getattr(out, "name", out) == old_name:
                if isinstance(out, str):
                    graph.outputs[i] = new_name
                else:
                    out.name = new_name
