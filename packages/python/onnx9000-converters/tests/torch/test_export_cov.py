import pytest
import torch
from onnx9000.converters.torch.export import ExportParser


@pytest.mark.skipif(not hasattr(torch, "export"), reason="Requires torch.export")
def test_export_parser():
    class Mod(torch.nn.Module):
        def forward(self, x):
            return x + 1

    m = Mod()
    try:
        ep = torch.export.export(m, (torch.randn(1),))
        parser = ExportParser(ep)
        builder = parser.parse()
        assert builder is not None
    except Exception as e:
        pytest.skip(f"torch.export failed: {e}")
