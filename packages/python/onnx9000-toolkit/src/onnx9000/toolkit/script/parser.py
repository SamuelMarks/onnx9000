"""Provide the AST parser that translates Python functions into ONNX graphs."""

from onnx9000.toolkit.script.builder import GraphBuilder


class ScriptParser:
    def __init__(self, globals_dict):
        self.builder = GraphBuilder()
        self.globals_dict = globals_dict

    def parse(self, func) -> GraphBuilder:
        return self.builder


def script(func):
    def wrapper(*args, **kwargs):
        parser = ScriptParser({})
        return parser.parse(func)  # mock build

    wrapper._is_onnx_script = True
    wrapper.to_builder = lambda: ScriptParser({}).parse(func)
    return wrapper
