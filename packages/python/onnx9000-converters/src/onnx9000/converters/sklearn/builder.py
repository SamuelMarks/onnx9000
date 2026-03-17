import typing
from typing import Any, Dict, List, Optional, Tuple, Union
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo, Attribute
from onnx9000.core.dtypes import DType


class SKLearnParser:
    """Parses a Scikit-Learn object into an ONNX Graph using pure duck-typing."""

    def __init__(
        self,
        model: object,
        name: str = "sklearn_model",
        initial_types: Optional[list[tuple]] = None,
    ) -> None:
        self.model = model
        self.graph = Graph(name)
        self.initial_types = initial_types or [("input", DType.FLOAT32, ("N", "C"))]

    def _is_type(self, obj: object, type_name: str) -> bool:
        """Zero dependency duck typing/class name checking."""
        return type(obj).__name__ == type_name

    def parse(self) -> Graph:
        input_name = self.initial_types[0][0]
        self.graph.inputs.append(
            ValueInfo(input_name, self.initial_types[0][1], self.initial_types[0][2])
        )
        output_names = self._parse_estimator(self.model, [input_name])
        for i, out_name in enumerate(output_names):
            self.graph.outputs.append(ValueInfo(out_name, DType.FLOAT32, ("N", "C")))
        return self.graph

    def _parse_estimator(self, estimator: object, input_names: list[str]) -> list[str]:
        """Parses an estimator and returns its output names."""
        est_type = type(estimator).__name__
        if est_type in ("GridSearchCV", "RandomizedSearchCV"):
            return self._parse_estimator(estimator.best_estimator_, input_names)
        if est_type == "Pipeline":
            current_inputs = input_names
            for _name, step in estimator.steps:
                if step != "passthrough":
                    current_inputs = self._parse_estimator(step, current_inputs)
            return current_inputs
        elif est_type == "FeatureUnion":
            parallel_outputs = []
            for _name, step in estimator.transformer_list:
                if step != "drop":
                    parallel_outputs.extend(self._parse_estimator(step, input_names))
            out_name = self.graph._uniquify_tensor_name("feature_union_concat")
            node = Node("Concat", domain="", inputs=parallel_outputs, outputs=[out_name])
            node.attrs["axis"] = Attribute("axis", "INT", 1)
            self.graph.nodes.append(node)
            return [out_name]
        elif est_type == "ColumnTransformer":
            parallel_outputs = []
            for _name, step, columns in estimator.transformers_:
                if step == "drop":
                    _ = None
                else:
                    extracted_out = self.graph._uniquify_tensor_name("col_extract")
                    col_indices_name = self.graph._uniquify_tensor_name("col_indices")
                    self.graph.tensors[col_indices_name] = Tensor(
                        col_indices_name, DType.INT64, (len(columns),)
                    )
                    extract_node = Node(
                        "ArrayFeatureExtractor",
                        domain="ai.onnx.ml",
                        inputs=[input_names[0], col_indices_name],
                        outputs=[extracted_out],
                    )
                    self.graph.nodes.append(extract_node)
                    if step == "passthrough":
                        parallel_outputs.append(extracted_out)
                    else:
                        step_out = self._parse_estimator(step, [extracted_out])
                        parallel_outputs.extend(step_out)
            if estimator.remainder == "passthrough":
                _ = None
            out_name = self.graph._uniquify_tensor_name("col_trans_concat")
            node = Node("Concat", domain="", inputs=parallel_outputs, outputs=[out_name])
            node.attrs["axis"] = Attribute("axis", "INT", 1)
            self.graph.nodes.append(node)
            return [out_name]
        elif est_type == "StandardScaler":
            out_name = self.graph._uniquify_tensor_name("scaler_out")
            node = Node("Scaler", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
            if hasattr(estimator, "mean_"):
                node.attrs["offset"] = Attribute("offset", "FLOATS", estimator.mean_.tolist())
            if hasattr(estimator, "scale_"):
                node.attrs["scale"] = Attribute("scale", "FLOATS", estimator.scale_.tolist())
            self.graph.nodes.append(node)
            return [out_name]
        elif est_type in ("LogisticRegression", "LinearSVC", "SGDClassifier"):
            out_label = self.graph._uniquify_tensor_name("label")
            out_prob = self.graph._uniquify_tensor_name("probabilities")
            node = Node(
                "LinearClassifier",
                domain="ai.onnx.ml",
                inputs=input_names,
                outputs=[out_label, out_prob],
            )
            node.attrs["coefficients"] = Attribute(
                "coefficients", "FLOATS", estimator.coef_.flatten().tolist()
            )
            node.attrs["intercepts"] = Attribute(
                "intercepts", "FLOATS", estimator.intercept_.tolist()
            )
            node.attrs["multi_class"] = Attribute(
                "multi_class", "INT", 1 if len(estimator.classes_) > 2 else 0
            )
            zipmap_out = self.graph._uniquify_tensor_name("zipmap_prob")
            zipmap_node = Node(
                "ZipMap", domain="ai.onnx.ml", inputs=[out_prob], outputs=[zipmap_out]
            )
            zipmap_node.attrs["classlabels_int64s"] = Attribute(
                "classlabels_int64s", "INTS", [int(c) for c in estimator.classes_]
            )
            self.graph.nodes.extend([node, zipmap_node])
            return [out_label, zipmap_out]
        else:
            out_name = self.graph._uniquify_tensor_name(f"{est_type.lower()}_out")
            node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
            self.graph.nodes.append(node)
            return [out_name]
