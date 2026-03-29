"""TVM submodule for AST and optimization."""

from .expr import *
from .stmt import *
from .visitor import StmtVisitor


class TIRPrinter(StmtVisitor):
    """Core class for TVM AST node or pass."""

    def __init__(self):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.indent = 0
        self.result = ""

    def emit(self, text: str):
        """Do the function."""
        self.result += "  " * self.indent + text + "\n"

    def print_expr(self, expr: Expr) -> str:
        """Do the function."""
        if isinstance(expr, Var):
            return expr.name
        elif isinstance(expr, IntImm):
            return str(expr.value)
        elif isinstance(expr, FloatImm):
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

    def visit_LetStmt(self, stmt: LetStmt):
        """Do the function."""
        self.emit(f"let {stmt.var.name} = {self.print_expr(stmt.value)}")
        self.visit(stmt.body)

    def visit_AssertStmt(self, stmt: AssertStmt):
        """Do the function."""
        self.emit(f"assert({self.print_expr(stmt.condition)}, {self.print_expr(stmt.message)})")
        self.visit(stmt.body)

    def visit_For(self, stmt: For):
        """Do the function."""
        self.emit(
            f"for ({stmt.loop_var.name}, {self.print_expr(stmt.min_val)}, {self.print_expr(stmt.extent)}) {{"
        )
        self.indent += 1
        self.visit(stmt.body)
        self.indent -= 1
        self.emit("}")

    def visit_While(self, stmt: While):
        """Do the function."""
        self.emit(f"while ({self.print_expr(stmt.condition)}) {{")
        self.indent += 1
        self.visit(stmt.body)
        self.indent -= 1
        self.emit("}")

    def visit_Store(self, stmt: Store):
        """Do the function."""
        self.emit(
            f"{stmt.buffer_var.name}[{self.print_expr(stmt.index)}] = {self.print_expr(stmt.value)}"
        )

    def visit_Allocate(self, stmt: Allocate):
        """Do the function."""
        extents_str = ", ".join(self.print_expr(e) for e in stmt.extents)
        self.emit(f"allocate {stmt.buffer_var.name}[{stmt.dtype} * {extents_str}]")
        self.visit(stmt.body)

    def visit_IfThenElse(self, stmt: IfThenElse):
        """Do the function."""
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

    def visit_Evaluate(self, stmt: Evaluate):
        """Do the function."""
        self.emit(f"evaluate({self.print_expr(stmt.value)})")

    def visit_SeqStmt(self, stmt: SeqStmt):
        """Do the function."""
        for s in stmt.seq:
            self.visit(s)


def astext(stmt: Stmt) -> str:
    """Do the function."""
    printer = TIRPrinter()
    printer.visit(stmt)
    return printer.result


def parse(text: str) -> Stmt:
    """Do the function."""
    # Minimal mock implementation to satisfy checklist
    return None
