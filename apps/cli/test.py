from onnx9000.converters.frontend.nn.module import Module


class MyModel(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
