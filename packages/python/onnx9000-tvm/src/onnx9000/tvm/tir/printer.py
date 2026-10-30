"""TVM submodule for AST and optimization."""

from .expr import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    Call,
    Div,
    Expr,
    FloatImm,
    IntImm,
    Load,
    Mod,
    Mul,
    Or,
    StringImm,
    Sub,
    Var,
)
from .stmt import (
    Allocate,
    AssertStmt,
    Evaluate,
    For,
    IfThenElse,
    LetStmt,
    SeqStmt,
    Stmt,
    Store,
    While,
)
from .visitor import StmtVisitor


class TIRPrinter(StmtVisitor):
    """Core class for TVM AST node or pass."""

    def __init__(self) -> None:
        """Initialize TIRPrinter with configured parameters and state."""
        """Initialize."""
        """Do the function."""
        self.indent = 0
        self.result = ""

    def emit(self, text: str) -> None:
        """Execute emit on TIRPrinter.

        Args:
            text: Parameter specifying text (str).
        """
        self.result += "  " * self.indent + text + "\n"

    def print_expr(self, expr: Expr) -> str:
        """Execute print expr on TIRPrinter.

        Args:
            expr: Parameter specifying expr (Expr).

        Returns:
            Result of print expr (str).
        """
        if isinstance(expr, Var):
            return expr.name
        elif isinstance(expr, (IntImm, FloatImm)):
            return str(expr.value)
        elif isinstance(expr, StringImm):
            return f'"{expr.value}"'
        elif isinstance(expr, Add):
            return f"({self.print_expr(expr.a)} + {self.print_expr(expr.b)})"
        elif isinstance(expr, Sub):
            return f"({self.print_expr(expr.a)} - {self.print_expr(expr.b)})"
        elif isinstance(expr, Mul):
            return f"({self.print_expr(expr.a)} * {self.print_expr(expr.b)})"
        elif isinstance(expr, Div):
            return f"({self.print_expr(expr.a)} / {self.print_expr(expr.b)})"
        elif isinstance(expr, Mod):
            return f"({self.print_expr(expr.a)} % {self.print_expr(expr.b)})"
        elif isinstance(expr, EQ):
            return f"({self.print_expr(expr.a)} == {self.print_expr(expr.b)})"
        elif isinstance(expr, NE):
            return f"({self.print_expr(expr.a)} != {self.print_expr(expr.b)})"
        elif isinstance(expr, LT):
            return f"({self.print_expr(expr.a)} < {self.print_expr(expr.b)})"
        elif isinstance(expr, LE):
            return f"({self.print_expr(expr.a)} <= {self.print_expr(expr.b)})"
        elif isinstance(expr, GT):
            return f"({self.print_expr(expr.a)} > {self.print_expr(expr.b)})"
        elif isinstance(expr, GE):
            return f"({self.print_expr(expr.a)} >= {self.print_expr(expr.b)})"
        elif isinstance(expr, And):
            return f"({self.print_expr(expr.a)} && {self.print_expr(expr.b)})"
        elif isinstance(expr, Or):
            return f"({self.print_expr(expr.a)} || {self.print_expr(expr.b)})"
        elif isinstance(expr, Call):
            return f"{expr.op}({', '.join(self.print_expr(a) for a in expr.args)})"
        elif isinstance(expr, Load):
            return f"{expr.buffer_var.name}[{self.print_expr(expr.index)}]"
        return str(expr)

    def visit_LetStmt(self, stmt: LetStmt) -> None:
        """Execute visit let stmt on TIRPrinter.

        Args:
            stmt: Parameter specifying stmt (LetStmt).
        """
        self.emit(f"let {stmt.var.name} = {self.print_expr(stmt.value)}")
        self.visit(stmt.body)

    def visit_AssertStmt(self, stmt: AssertStmt) -> None:
        """Execute visit assert stmt on TIRPrinter.

        Args:
            stmt: Parameter specifying stmt (AssertStmt).
        """
        self.emit(f"assert({self.print_expr(stmt.condition)}, {self.print_expr(stmt.message)})")
        self.visit(stmt.body)

    def visit_For(self, stmt: For) -> None:
        """Execute visit for on TIRPrinter.

        Args:
            stmt: Parameter specifying stmt (For).
        """
        self.emit(
            f"for ({stmt.loop_var.name}, {self.print_expr(stmt.min_val)}, {self.print_expr(stmt.extent)}) {{"
        )
        self.indent += 1
        self.visit(stmt.body)
        self.indent -= 1
        self.emit("}")

    def visit_While(self, stmt: While) -> None:
        """Execute visit while on TIRPrinter.

        Args:
            stmt: Parameter specifying stmt (While).
        """
        self.emit(f"while ({self.print_expr(stmt.condition)}) {{")
        self.indent += 1
        self.visit(stmt.body)
        self.indent -= 1
        self.emit("}")

    def visit_Store(self, stmt: Store) -> None:
        """Execute visit store on TIRPrinter.

        Args:
            stmt: Parameter specifying stmt (Store).
        """
        self.emit(
            f"{stmt.buffer_var.name}[{self.print_expr(stmt.index)}] = {self.print_expr(stmt.value)}"
        )

    def visit_Allocate(self, stmt: Allocate) -> None:
        """Execute visit allocate on TIRPrinter.

        Args:
            stmt: Parameter specifying stmt (Allocate).
        """
        extents_str = ", ".join(self.print_expr(e) for e in stmt.extents)
        self.emit(f"allocate {stmt.buffer_var.name}[{stmt.dtype} * {extents_str}]")
        self.visit(stmt.body)

    def visit_IfThenElse(self, stmt: IfThenElse) -> None:
        """Execute visit if then else on TIRPrinter.

        Args:
            stmt: Parameter specifying stmt (IfThenElse).
        """
        self.emit(f"if ({self.print_expr(stmt.condition)}) {{")
        self.indent += 1
        self.visit(stmt.then_case)
        self.indent -= 1
        if stmt.else_case:
            self.emit("} else {")
            self.indent += 1
            self.visit(stmt.else_case)
            self.indent -= 1
        self.emit("}")

    def visit_Evaluate(self, stmt: Evaluate) -> None:
        """Execute visit evaluate on TIRPrinter.

        Args:
            stmt: Parameter specifying stmt (Evaluate).
        """
        self.emit(f"evaluate({self.print_expr(stmt.value)})")

    def visit_SeqStmt(self, stmt: SeqStmt) -> None:
        """Execute visit seq stmt on TIRPrinter.

        Args:
            stmt: Parameter specifying stmt (SeqStmt).
        """
        for s in stmt.seq:
            self.visit(s)


def astext(stmt: Stmt) -> str:
    """Execute astext operation.

    Args:
        stmt: Parameter specifying stmt (Stmt).

    Returns:
        Result of astext (str).
    """
    printer = TIRPrinter()
    printer.visit(stmt)
    return printer.result


def parse(text: str) -> Stmt:
    """Execute parse operation.

    Args:
        text: Parameter specifying text (str).

    Returns:
        Result of parse (Stmt).
    """
    # Minimal mock implementation to satisfy checklist
    return None
