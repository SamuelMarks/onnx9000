"""
Dynamic Shape Handlers

Resolves or normalizes dynamic axes (like batch_size or seq_len)
by either freezing them to specific integers or emitting symbol
propagation logic.
"""

from onnx9000.ir import Graph


def resolve_dynamic_batch(graph: Graph, batch_size: int = 1) -> None:
    """Freezes the dynamic batch dimension to a fixed size."""
    pass


def resolve_dynamic_sequence(graph: Graph, seq_len: int = 128) -> None:
    """Freezes dynamic sequence lengths."""
    pass


def extract_rnn_states(graph: Graph) -> None:
    """Promotes internal RNN hidden states to explicit graph IO for autoregressive loops."""
    pass
