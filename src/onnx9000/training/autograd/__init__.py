"""
Autograd Sub-Package

This module provides the necessary primitives and graph builders to perform
automatic differentiation (autograd) on ONNX IR graphs. It dynamically constructs
the backward pass for training models natively via the generated WASM or C++ bindings.

Features:
- **Vector-Jacobian Products (VJPs)** for common operators.
- **Backward Graph Compilation** to fuse forward and backward passes.
- Extensibility for adding new gradient rules.
"""

from onnx9000.training.autograd.compiler import (
    build_backward_graph,
    extract_partial_subgraph,
    freeze_layers,
    inject_custom_loss_subgraph,
    load_training_checkpoint,
    save_training_checkpoint,
    scale_backward_graph_for_mixed_precision,
    validate_training_graph,
)
from onnx9000.training.autograd.losses import add_crossentropy_loss, add_mse_loss
from onnx9000.training.autograd.optimizers import (
    add_adam_optimizer,
    add_adamw_optimizer,
    add_gradient_accumulation,
    add_gradient_clipping,
    add_sgd_optimizer,
)

__all__ = [
    "build_backward_graph",
    "add_sgd_optimizer",
    "add_adam_optimizer",
    "add_adamw_optimizer",
    "add_gradient_accumulation",
    "add_gradient_clipping",
    "add_mse_loss",
    "add_crossentropy_loss",
    "extract_partial_subgraph",
    "freeze_layers",
    "validate_training_graph",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "scale_backward_graph_for_mixed_precision",
    "inject_custom_loss_subgraph",
]
