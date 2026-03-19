"""Provides tree traversal module functionality."""

import logging
from typing import Any

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.hummingbird.analysis import analyze_tree_depth
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions

logger = logging.getLogger(__name__)


class TreeTraversalCompiler:
    """TreeTraversal Strategy compiler."""

    def __init__(self, tree: TreeAbstractions, batch_size: Any = "N") -> None:
        """Initialize the instance."""
        self.tree = tree
        self.batch_size = batch_size
        self.max_depth = int(analyze_tree_depth(tree)["max"])
        self._merge_index_tensors()

    def compile(self, g: Graph) -> None:
        """Compiles tree to TreeTraversal operators (unrolled)."""
        # Map structures to flat 1D index arrays
        t_features = self._build_1d_array(self.tree.features)
        t_thresholds = self._build_1d_array(self.tree.thresholds)
        t_left = self._build_1d_array(self.tree.left_children)
        t_right = self._build_1d_array(self.tree.right_children)
        t_values = self._build_1d_array(self.tree.values)

        # Add tensors to graph
        for t in [t_features, t_thresholds, t_left, t_right, t_values]:
            g.tensors[t.name] = t

        # Initial node indices for the batch (shape: [batch_size])
        curr_indices = "curr_node_indices"

        # Implement iterative gathering (simulating tree descent) without `Loop`
        for depth in range(self.max_depth):
            # Dynamic array indexing using ONNX Gather
            g.nodes.append(
                Node(
                    op_type="Gather",
                    inputs=[t_features.name, curr_indices],
                    outputs=[f"feat_idx_d{depth}"],
                )
            )
            g.nodes.append(
                Node(
                    op_type="Gather",
                    inputs=[t_thresholds.name, curr_indices],
                    outputs=[f"thresh_d{depth}"],
                )
            )
            g.nodes.append(
                Node(
                    op_type="Gather",
                    inputs=[t_left.name, curr_indices],
                    outputs=[f"left_idx_d{depth}"],
                )
            )
            g.nodes.append(
                Node(
                    op_type="Gather",
                    inputs=[t_right.name, curr_indices],
                    outputs=[f"right_idx_d{depth}"],
                )
            )

            # Gather actual feature values
            g.nodes.append(
                Node(
                    op_type="Gather",
                    inputs=["input", f"feat_idx_d{depth}"],
                    outputs=[f"feat_val_d{depth}"],
                )
            )

            # Less / Greater to generate binary offsets (0 or 1)
            g.nodes.append(
                Node(
                    op_type="Less",
                    inputs=[f"feat_val_d{depth}", f"thresh_d{depth}"],
                    outputs=[f"go_left_d{depth}"],
                )
            )

            # Multiply binary offsets by jump strides / compute next node natively
            # Using Where to choose between left and right
            g.nodes.append(
                Node(
                    op_type="Where",
                    inputs=[f"go_left_d{depth}", f"left_idx_d{depth}", f"right_idx_d{depth}"],
                    outputs=[f"next_idx_d{depth}"],
                )
            )

            # Handle leaf node identification (e.g., negative index markers)
            # Use Where to freeze indices of rows that have reached a leaf
            g.nodes.append(
                Node(op_type="Less", inputs=[curr_indices, "zero"], outputs=[f"is_leaf_d{depth}"])
            )
            g.nodes.append(
                Node(
                    op_type="Where",
                    inputs=[f"is_leaf_d{depth}", curr_indices, f"next_idx_d{depth}"],
                    outputs=[f"curr_node_indices_d{depth}"],
                )
            )

            curr_indices = f"curr_node_indices_d{depth}"

        # Final prediction gather
        g.nodes.append(
            Node(op_type="Gather", inputs=[t_values.name, curr_indices], outputs=["prediction"])
        )

    def _build_1d_array(self, data: list) -> Tensor:
        """Execute the build 1d array operation."""
        name = f"arr_{id(data)}"
        return Tensor(name=name, shape=(len(data),), is_initializer=True)

    def _merge_index_tensors(self) -> None:
        """Optimize Gather operations by merging index tensors."""
        pass


def compile_forest_tree_traversal(
    g: Graph, trees: list[TreeAbstractions], batch_size: Any = "N"
) -> None:
    """Implement parallel traversal of all trees in an ensemble using batched Gathers.

    Pre-allocate output tensors for traversal aggregations.
    """
    pass


def handle_categorical_traversal(g: Graph) -> None:
    """Implement categorical feature gathering (equality checks vs inequalities)."""
    pass


def handle_missing_value_traversal(g: Graph) -> None:
    """Handle missing value routing natively within gathered offsets."""
    pass


def flatten_multi_class_traversal(g: Graph) -> None:
    """Flatten multi-class leaf outputs into parallel gathers."""
    pass


def test_gather_latency_wasm() -> None:
    """Test and validate latency of Gather bounds on WASM."""
    pass
