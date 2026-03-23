"""AST Builder for generating strictly compliant C89 source code."""

import re


class C89Builder:
    def __init__(self, prefix: str = ""):
        self.code: list[str] = []
        self.indent_level = 0
        self.prefix = prefix
        self.var_counters = {}

    def _sanitize(self, name: str) -> str:
        """Translate ONNX names to valid C identifiers natively."""
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if sanitized and sanitized[0].isdigit():
            sanitized = "v" + sanitized
        return sanitized

    def emit(self, text: str) -> None:
        if text.strip():
            self.code.append("    " * self.indent_level + text)
        else:
            self.code.append("")

    def emit_comment(self, text: str) -> None:
        """Emit C89 compliant comment (/* */)."""
        self.emit(f"/* {text} */")

    def emit_include(self, header: str, is_system: bool = True) -> None:
        if is_system:
            self.emit(f"#include <{header}>")
        else:
            self.emit(f'#include "{header}"')

    def push_indent(self) -> None:
        self.indent_level += 1

    def pop_indent(self) -> None:
        self.indent_level -= 1
        if self.indent_level < 0:
            self.indent_level = 0

    def get_code(self) -> str:
        return "\n".join(self.code) + "\n"

    def new_var(self, name: str) -> str:
        """Generate a unique variable name (e.g., i_1)."""
        if name not in self.var_counters:
            self.var_counters[name] = 0
            return name
        self.var_counters[name] += 1
        return f"{name}_{self.var_counters[name]}"
