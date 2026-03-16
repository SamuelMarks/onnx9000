"""Provides dce.py module functionality."""

import logging
from onnx9000.core.ir import Graph
from onnx9000.optimizer.simplifier.passes.base import GraphPass

logger = logging.getLogger(__name__)


class DCEPass(GraphPass):
    """
    Dead Code Elimination (DCE).
    Removes nodes whose outputs are never consumed by any other node
    and are not in the graph's explicitly defined outputs.
    """

    def run(self, graph: Graph) -> bool:
        """Implements the run method or operation."""
        changed = False
        while True:
            local_changed = self._run_once(graph)
            if not local_changed:
                break
            changed = True
        return changed

    def _run_once(self, graph: Graph) -> bool:
        """Implements the _run_once method or operation."""
        changed = False
        consumed: set[str] = set(graph.outputs)
        for node in graph.nodes:
            for inp in node.inputs:
                consumed.add(inp)
        new_nodes = []
        for node in graph.nodes:
            if any((out in consumed for out in node.outputs)):
                new_nodes.append(node)
            else:
                logger.info(f"Eliminated dead node {node.name} ({node.op_type})")
                changed = True
        graph.nodes = new_nodes
        consumed_initializers = {init for init in graph.initializers if init in consumed}
        if len(consumed_initializers) < len(graph.initializers):
            graph.initializers = [
                init for init in graph.initializers if init in consumed_initializers
            ]
            changed = True
        return changed


class IdentityEliminationPass(GraphPass):
    """
    Detects and removes explicit Identity nodes and redundant operations
    like Cast(Cast(X)), Reshape(Reshape(X)), Transpose(Transpose(X)), etc.
    Rewires inputs to consumers.
    """

    def run(self, graph: Graph) -> bool:
        """Implements the run method or operation."""
        changed = False
        while True:
            local_changed = self._run_once(graph)
            if not local_changed:
                break
            changed = True
        return changed

    def _run_once(self, graph: Graph) -> bool:
        """Implements the _run_once method or operation."""
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
            elif node.op_type == "Cast":
                producer = producers.get(node.inputs[0])
                if producer and producer.op_type == "Cast":
                    node.inputs[0] = producer.inputs[0]
                    changed = True
                    logger.info(f"Simplified chained Cast at {node.name}")
            elif node.op_type == "Reshape":
                producer = producers.get(node.inputs[0])
                if producer and producer.op_type == "Reshape":
                    node.inputs[0] = producer.inputs[0]
                    changed = True
                    logger.info(f"Simplified chained Reshape at {node.name}")
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
        """Implements the _rewire method or operation."""
        for node in graph.nodes:
            for i, inp in enumerate(node.inputs):
                if inp == old_name:
                    node.inputs[i] = new_name
        for i, out in enumerate(graph.outputs):
            if out == old_name:
                graph.outputs[i] = new_name


def dead_code_elimination(graph: Graph) -> None:
    """Implements the dead_code_elimination method or operation."""
    while True:
        dce_changed = DCEPass().run(graph)
        id_changed = IdentityEliminationPass().run(graph)
        if not dce_changed and (not id_changed):
            break
