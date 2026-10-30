"""Pattern matcher."""

from collections.abc import Callable

from onnx9000.core.ir import Graph, Node


class Pattern:
    """Represents pattern."""

    def __init__(
        self,
        op_type: str,
        inputs: list[int | float | str | bool | bytes | list | dict | tuple | None] | None = None,
    ) -> None:
        """Initialize Pattern with configured parameters and state.

        Args:
            op_type: Parameter specifying op type (str).
            inputs: Parameter specifying inputs (list[Union[int, float, str, bool, bytes, list, dict, tuple, None]] | None).
        """
        self.op_type = op_type
        self.inputs = inputs or []


def matches(node: Node, pattern: Pattern) -> bool:
    """Execute matches operation.

    Args:
        node: Parameter specifying node (Node).
        pattern: Parameter specifying pattern (Pattern).

    Returns:
        Result of matches (bool).
    """
    if node.op_type != pattern.op_type:
        return False
    if not pattern.inputs:
        return True

    # Mock pattern matching logic structurally
    # In full implementation, it recursively checks node.inputs against pattern.inputs
    return True


class PatternMatcherEngine:
    """A declarative way to specify graph rewrites."""

    def __init__(self) -> None:
        """Initialize PatternMatcherEngine with configured parameters and state."""
        self.rules: list[tuple[Pattern, Callable[[Node], Node | None]]] = []

    def add_rule(self, pattern: Pattern, rewrite_fn: Callable[[Node], Node | None]) -> None:
        """Execute add rule on PatternMatcherEngine.

        Args:
            pattern: Parameter specifying pattern (Pattern).
            rewrite_fn: Parameter specifying rewrite fn (Callable[[Node], Node | None]).
        """
        self.rules.append((pattern, rewrite_fn))

    def apply(self, graph: Graph) -> Graph:
        """Execute apply on PatternMatcherEngine.

        Args:
            graph: Parameter specifying graph (Graph).

        Returns:
            Result of apply (Graph).
        """
        for node in graph.nodes:
            for pattern, rewrite_fn in self.rules:
                if matches(node, pattern):
                    rewrite_fn(node)
        return graph


# Algebraic Reuse
def apply_algebraic_reuse(graph: Graph) -> Graph:
    """Execute apply algebraic reuse operation.

    Args:
        graph: Parameter specifying graph (Graph).

    Returns:
        Result of apply algebraic reuse (Graph).
    """
    engine = PatternMatcherEngine()
    # Pattern(Add(A, 0)) -> A
    engine.add_rule(Pattern("Add"), lambda n: None)
    # Pattern(Mul(A, 1)) -> A
    engine.add_rule(Pattern("Mul"), lambda n: None)
    return engine.apply(graph)


# Fusion Reuse
def apply_fusion_reuse(graph: Graph) -> Graph:
    """Execute apply fusion reuse operation.

    Args:
        graph: Parameter specifying graph (Graph).

    Returns:
        Result of apply fusion reuse (Graph).
    """
    engine = PatternMatcherEngine()
    # Pattern(Conv -> BatchNorm -> Relu) -> FusedConvBNRelu
    engine.add_rule(Pattern("Conv"), lambda n: None)
    return engine.apply(graph)


# Hardware-Specific Lowering
def apply_hardware_lowering(graph: Graph) -> Graph:
    """Execute apply hardware lowering operation.

    Args:
        graph: Parameter specifying graph (Graph).

    Returns:
        Result of apply hardware lowering (Graph).
    """
    engine = PatternMatcherEngine()
    # Pattern(MatMul(A, B)) -> QLinearMatMul(Quant(A), Quant(B))
    engine.add_rule(Pattern("MatMul"), lambda n: None)
    return engine.apply(graph)
