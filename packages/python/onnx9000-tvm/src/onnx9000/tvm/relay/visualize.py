"""TVM submodule for AST and optimization."""

from .analysis import topological_sort
from .expr import Call, Constant, Expr, Function, If, Let, Op, TupleExpr, TupleGetItem, Var
from .visitor import ExprVisitor


class DotPrinter(ExprVisitor):
    """Core class for TVM AST node or pass."""

    def __init__(self):
        """Initialize the DOT printer with a header and empty node mapping."""
        self.result = "digraph IR {\n"
        self.node_ids = {}

    def get_id(self, expr: Expr) -> str:
        """Get or create a unique identifier for an expression node."""
        if id(expr) not in self.node_ids:
            self.node_ids[id(expr)] = f"node_{len(self.node_ids)}"
        return self.node_ids[id(expr)]

    def _add_edge(self, src: Expr, dst: Expr, label: str = ""):
        """Add an edge between two nodes in the DOT representation."""
        self.result += f'  {self.get_id(src)} -> {self.get_id(dst)} [label="{label}"];\n'

    def visit(self, expr: Expr) -> None:
        """Perform a topological traversal and generate DOT nodes and edges."""
        nodes = topological_sort(expr)

        for node in nodes:
            node_id = self.get_id(node)
            label = ""
            if isinstance(node, Var):
                label = f"Var(%{node.name_hint})"
            elif isinstance(node, Constant):
                label = "Constant"
            elif isinstance(node, Op):
                label = f"Op({node.name})"
            elif isinstance(node, Call):
                label = "Call"
                self._add_edge(node, node.op, "op")
                for i, arg in enumerate(node.args):
                    self._add_edge(node, arg, f"arg_{i}")
            elif isinstance(node, TupleExpr):
                label = "tuple"
                for i, f in enumerate(node.fields):
                    self._add_edge(node, f, f"field_{i}")
            elif isinstance(node, TupleGetItem):
                label = f"TupleGetItem({node.index})"
                self._add_edge(node, node.tuple_value, "tuple")
            elif isinstance(node, Let):
                label = "Let"
                self._add_edge(node, node.var, "var")
                self._add_edge(node, node.value, "value")
                self._add_edge(node, node.body, "body")
            elif isinstance(node, If):
                label = "If"
                self._add_edge(node, node.cond, "cond")
                self._add_edge(node, node.true_branch, "true")
                self._add_edge(node, node.false_branch, "false")
            elif isinstance(node, Function):
                label = "Function"
                for i, p in enumerate(node.params):
                    self._add_edge(node, p, f"param_{i}")
                self._add_edge(node, node.body, "body")

            self.result += f'  {node_id} [label="{label}"];\n'

        self.result += "}\n"


def to_dot(expr: Expr) -> str:
    """Return a Graphviz DOT representation of the AST."""
    printer = DotPrinter()
    printer.visit(expr)
    return printer.result
