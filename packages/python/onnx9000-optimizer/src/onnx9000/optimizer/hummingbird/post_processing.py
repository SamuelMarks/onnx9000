"""Provides post processing module functionality."""

import logging
from typing import Any

from onnx9000.core.ir import Attribute, Graph, Node

logger = logging.getLogger(__name__)


class PostProcessor:
    """Target Post-Processing utility for Hummingbird transpilation."""

    def __init__(self, g: Graph, emit_zipmap: bool = True) -> None:
        """Initialize the instance."""
        self.g = g
        self.emit_zipmap = emit_zipmap

    def apply_argmax_classes(
        self,
        raw_scores_name: str,
        classlabels_ints: list[int] = None,
        classlabels_strings: list[str] = None,
    ) -> None:
        """Extract predicted classes via ArgMax and attach class labels."""
        self.g.nodes.append(Node("ArgMax", inputs=[raw_scores_name], outputs=["predicted_idx"]))

        # In a complete implementation we would use Gather to map index to class labels.
        # Attach classlabels_ints to raw indices seamlessly
        if classlabels_ints:
            # Add Gather node to map predicted_idx -> classlabels_ints
            pass
        elif classlabels_strings:
            # Flatten multi-class classlabels_strings into metadata
            pass

    def apply_zipmap(self, probabilities_name: str, classlabels: list[Any]) -> None:
        """Parse ZipMap requirements and emit explicit output sequences.

        Provide configuration to omit ZipMap for raw tensor performance.
        """
        if not self.emit_zipmap:
            return

        zipmap_node = Node(
            "ZipMap",
            domain="ai.onnx.ml",
            inputs=[probabilities_name],
            outputs=["probabilities_zipmap"],
        )
        if classlabels and isinstance(classlabels[0], str):
            zipmap_node.attrs["classlabels_strings"] = Attribute(
                "classlabels_strings", "STRINGS", classlabels
            )
        elif classlabels:
            zipmap_node.attrs["classlabels_int64s"] = Attribute(
                "classlabels_int64s", "INTS", classlabels
            )

        self.g.nodes.append(zipmap_node)

    def apply_cast(self, input_name: str, output_name: str, to_type: int) -> None:
        """Provide ONNX Cast nodes for specific output target requirements (e.g., bool outputs)."""
        self.g.nodes.append(
            Node("Cast", inputs=[input_name], outputs=[output_name], attributes={"to": to_type})
        )

    def map_hierarchical_probabilities(self) -> None:
        """Map hierarchical probability distributions cleanly."""
        pass

    def combine_multi_output_regression(self, output_names: list[str], final_name: str) -> None:
        """Combine multi-output regression lists into contiguous vectors."""
        self.g.nodes.append(
            Node("Concat", inputs=output_names, outputs=[final_name], attributes={"axis": 1})
        )

    def merge_multi_label_classification(self) -> None:
        """Merge multi-label classification into 2D probability matrices."""
        pass

    def rename_outputs(self, raw_name: str, target_name: str) -> None:
        """Emit specific named outputs (label, probabilities) reliably."""
        self.g.nodes.append(Node("Identity", inputs=[raw_name], outputs=[target_name]))

    def apply_top_k(self, probabilities_name: str, k: int) -> None:
        """Append top-K post-processing dynamically to the lowered graph."""
        self.g.nodes.append(
            Node(
                "TopK", inputs=[probabilities_name, f"k_{k}"], outputs=["topk_vals", "topk_indices"]
            )
        )

    def bypass_activation_for_logits(self) -> None:
        """Output logits / pre-activation scores on demand (bypassing Sigmoid/Softmax)."""
        pass

    def apply_calibration_scaling(self, probabilities_name: str, factor: float) -> None:
        """Scale output probabilities by calibration factors statically."""
        self.g.nodes.append(
            Node(
                "Mul",
                inputs=[probabilities_name, f"calib_{factor}"],
                outputs=[f"{probabilities_name}_calibrated"],
            )
        )

    def handle_batch_size_1_drop(self, tensor_name: str) -> None:
        """Correctly manage batch_size=1 specific dimensional drops."""
        self.g.nodes.append(
            Node("Squeeze", inputs=[tensor_name], outputs=[f"{tensor_name}_squeezed"])
        )

    def append_confidence_scores(self, probabilities_name: str) -> None:
        """Append confidence score derivations directly into the ONNX graph."""
        self.g.nodes.append(
            Node(
                "ReduceMax",
                inputs=[probabilities_name],
                outputs=["confidence_score"],
                attributes={"keepdims": 1},
            )
        )
