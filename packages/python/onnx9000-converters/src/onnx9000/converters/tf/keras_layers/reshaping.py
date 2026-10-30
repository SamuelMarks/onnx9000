"""Keras reshaping, dense, and core structural layer converters."""

from collections.abc import Callable

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_keras_dense(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Dense operation to ONNX equivalent."""
    if len(node.inputs) == 3:
        mm = builder.make_node("MatMul", [node.inputs[0], node.inputs[1]], {}, f"{node.name}_mm")[0]
        return builder.make_node("Add", [mm, node.inputs[2]], node.attr, node.name)
    return builder.make_node("MatMul", node.inputs[:2], node.attr, node.name)


def _map_keras_flatten(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Flatten operation to ONNX equivalent."""
    return builder.make_node("Flatten", node.inputs, node.attr, node.name)


def _map_keras_reshape(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Reshape operation to ONNX equivalent."""
    return builder.make_node("Reshape", node.inputs, node.attr, node.name)


def _map_keras_permute(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Permute operation to ONNX equivalent."""
    return builder.make_node("Transpose", node.inputs, node.attr, node.name)


def _map_keras_repeat_vector(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Repeat Vector operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    n = int(node.attr.get("n", 1))
    axes_name = builder.make_node("Constant", [], {"value": [1]}, node.name + "_axes")[0]
    unsqueezed = builder.make_node(
        "Unsqueeze", [node.inputs[0], axes_name], {}, node.name + "_unsqueezed"
    )[0]
    repeats_name = builder.make_node("Constant", [], {"value": [1, n, 1]}, node.name + "_repeats")[
        0
    ]
    return builder.make_node("Tile", [unsqueezed, repeats_name], {}, node.name)


def _map_keras_cropping(dim: str) -> Callable:
    """Map Keras Cropping operation to ONNX equivalent."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the impl operation."""
        if not node.inputs:
            return builder.make_node("Constant", [], node.attr, node.name)
        attr = dict(node.attr)
        cropping = attr.pop("cropping", None)
        if cropping is None:
            if dim == "1D":
                cropping = (1, 1)
            elif dim == "2D":
                cropping = ((1, 1), (1, 1))
            else:
                cropping = ((1, 1), (1, 1), (1, 1))
        if isinstance(cropping, int):
            if dim == "1D":
                cropping = (cropping, cropping)
            elif dim == "2D":
                cropping = ((cropping, cropping), (cropping, cropping))
            else:
                cropping = (
                    (cropping, cropping),
                    (cropping, cropping),
                    (cropping, cropping),
                )
        starts = []
        ends = []
        axes = []
        if dim == "1D":
            starts.append(cropping[0])
            ends.append(-cropping[1] if cropping[1] != 0 else 9223372036854775807)
            axes.append(2)
        elif dim == "2D":
            if isinstance(cropping[0], int):
                cropping = ((cropping[0], cropping[0]), (cropping[1], cropping[1]))
            starts.extend([cropping[0][0], cropping[1][0]])
            ends.extend(
                [
                    -cropping[0][1] if cropping[0][1] != 0 else 9223372036854775807,
                    -cropping[1][1] if cropping[1][1] != 0 else 9223372036854775807,
                ]
            )
            axes.extend([2, 3])
        else:
            if isinstance(cropping[0], int):
                cropping = (
                    (cropping[0], cropping[0]),
                    (cropping[1], cropping[1]),
                    (cropping[2], cropping[2]),
                )
            starts.extend([cropping[0][0], cropping[1][0], cropping[2][0]])
            ends.extend(
                [
                    -cropping[0][1] if cropping[0][1] != 0 else 9223372036854775807,
                    -cropping[1][1] if cropping[1][1] != 0 else 9223372036854775807,
                    -cropping[2][1] if cropping[2][1] != 0 else 9223372036854775807,
                ]
            )
            axes.extend([2, 3, 4])
        starts_name = builder.make_node("Constant", [], {"value": starts}, node.name + "_starts")[0]
        ends_name = builder.make_node("Constant", [], {"value": ends}, node.name + "_ends")[0]
        axes_name = builder.make_node("Constant", [], {"value": axes}, node.name + "_axes")[0]
        return builder.make_node(
            "Slice",
            [node.inputs[0], starts_name, ends_name, axes_name],
            attr,
            node.name,
        )

    return _impl


def _map_keras_upsampling(dim: str) -> Callable:
    """Map Keras Upsampling operation to ONNX equivalent."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the impl operation."""
        if not node.inputs:
            return builder.make_node("Constant", [], node.attr, node.name)
        attr = dict(node.attr)
        size = attr.pop("size", None)
        interpolation = attr.pop("interpolation", "nearest")
        if isinstance(interpolation, bytes):
            interpolation = interpolation.decode("utf-8")
        if interpolation == "bilinear":
            interpolation = "linear"
        attr["mode"] = interpolation.encode("utf-8")
        if size is None:
            if dim == "1D":
                size = [2]
            elif dim == "2D":
                size = [2, 2]
            else:
                size = [2, 2, 2]
        if isinstance(size, int):
            size = [size]
        scales = [1.0, 1.0] + [float(s) for s in size]
        roi_name = builder.make_node("Constant", [], {"value": []}, node.name + "_roi")[0]
        scales_name = builder.make_node("Constant", [], {"value": scales}, node.name + "_scales")[0]
        return builder.make_node("Resize", [node.inputs[0], roi_name, scales_name], attr, node.name)

    return _impl


def _map_keras_layers_tfsm_layer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Tfsm Layer operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Deserialize operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Serialize operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__dense(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Dense operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_einsum_dense(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Einsum Dense operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    equation = node.attr.get("equation", "")
    if isinstance(equation, str):
        equation = equation.encode("utf-8")
    if len(node.inputs) > 2:
        einsum = builder.make_node(
            "Einsum", node.inputs[:2], {"equation": equation}, node.name + "_einsum"
        )[0]
        return builder.make_node("Add", [einsum, node.inputs[2]], {}, node.name)
    return builder.make_node("Einsum", node.inputs[:2], {"equation": equation}, node.name)


def _map_keras_layers__einsum_dense(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Einsum Dense operation to ONNX equivalent."""
    return _map_keras_einsum_dense(builder, node)


def _map_keras_embedding(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Embedding operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    if len(node.inputs) == 2:
        return builder.make_node("Gather", [node.inputs[1], node.inputs[0]], attr, node.name)
    else:
        return builder.make_node("Gather", node.inputs, attr, node.name)


def _map_keras_layers__identity(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Identity operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__input(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Input operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__input_layer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Input Layer operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__lambda(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Lambda operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    func_str = node.attr.get("function", "")
    if isinstance(func_str, bytes):
        func_str = func_str.decode("utf-8")
    if "backend.sqrt" in func_str or "tf.sqrt" in func_str:
        return builder.make_node("Sqrt", node.inputs[:1], {}, node.name)
    if "backend.exp" in func_str or "tf.exp" in func_str:
        return builder.make_node("Exp", node.inputs[:1], {}, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_masking(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Masking operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    mask_value = node.attr.get("mask_value", 0.0)
    val_node = builder.make_node(
        "Constant", [], {"value": float(mask_value)}, node.name + "_mask_val"
    )[0]
    is_mask_val = builder.make_node("Equal", [node.inputs[0], val_node], {}, node.name + "_eq")[0]
    not_mask_val = builder.make_node("Not", [is_mask_val], {}, node.name + "_mask")[0]
    cast_float = builder.make_node("Cast", [not_mask_val], {"to": 1}, node.name + "_cast_float")[0]
    axes_node = builder.make_node("Constant", [], {"value": [-1]}, node.name + "_axes")[0]
    reduced = builder.make_node(
        "ReduceMax", [cast_float, axes_node], {"keepdims": 0}, node.name + "_reduced"
    )[0]
    final_mask = builder.make_node("Cast", [reduced], {"to": 9}, node.name + "_bool_mask")[0]
    main_out = builder.make_node("Identity", [node.inputs[0]], node.attr, node.name)[0]
    return [main_out, final_mask]


def _map_keras_layers__reversible_embedding(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Reversible Embedding operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__wrapper(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Wrapper operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__input_spec(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Input Spec operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__layer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Layer operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__cropping1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Cropping1 D operation to ONNX equivalent."""
    return _map_keras_cropping("1D")(builder, node)


def _map_keras_layers__cropping2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Cropping2 D operation to ONNX equivalent."""
    return _map_keras_cropping("2D")(builder, node)


def _map_keras_layers__cropping3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Cropping3 D operation to ONNX equivalent."""
    return _map_keras_cropping("3D")(builder, node)


def _map_keras_layers__flatten(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Flatten operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__permute(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Permute operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__repeat_vector(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Repeat Vector operation to ONNX equivalent."""
    return _map_keras_repeat_vector(builder, node)


def _map_keras_layers__reshape(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Reshape operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__up_sampling1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Up Sampling1 D operation to ONNX equivalent."""
    return _map_keras_upsampling("1D")(builder, node)


def _map_keras_layers__up_sampling2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Up Sampling2 D operation to ONNX equivalent."""
    return _map_keras_upsampling("2D")(builder, node)


def _map_keras_layers__up_sampling3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Up Sampling3 D operation to ONNX equivalent."""
    return _map_keras_upsampling("3D")(builder, node)


def _map_keras_layers__flax_layer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Flax Layer operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__jax_layer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Jax Layer operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__torch_module_wrapper(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Torch Module Wrapper operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__embedding(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Embedding operation to ONNX equivalent."""
    return _map_keras_embedding(builder, node)


def _map_keras_layers__masking(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Masking operation to ONNX equivalent."""
    return _map_keras_masking(builder, node)
