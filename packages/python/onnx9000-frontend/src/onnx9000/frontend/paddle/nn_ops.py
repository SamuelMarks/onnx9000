"""Module docstring."""

from typing import Callable
from onnx9000.frontend.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.frontend.paddle.parsers import PaddleNode


def _map_matmul(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map matmul operation."""
    x = node.inputs.get("X", [])
    y = node.inputs.get("Y", [])
    trans_x = builder.extract_attr(node, "transpose_X", False)
    trans_y = builder.extract_attr(node, "transpose_Y", False)
    x_in = x[0] if x else ""
    y_in = y[0] if y else ""
    if trans_x:
        x_in = builder.make_node("Transpose", [x_in], {"perm": [1, 0]}, f"{node.name}_tx")[0]
    if trans_y:
        y_in = builder.make_node("Transpose", [y_in], {"perm": [1, 0]}, f"{node.name}_ty")[0]
    return builder.make_node("MatMul", [x_in, y_in], {}, node.name)


def _map_mul(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map mul operation."""
    x = node.inputs.get("X", [])
    y = node.inputs.get("Y", [])
    x_num_col_dims = builder.extract_attr(node, "x_num_col_dims", 1)
    y_num_col_dims = builder.extract_attr(node, "y_num_col_dims", 1)
    x_flatten = builder.make_node("Flatten", x, {"axis": x_num_col_dims}, f"{node.name}_x_flatten")[
        0
    ]
    y_flatten = builder.make_node("Flatten", y, {"axis": y_num_col_dims}, f"{node.name}_y_flatten")[
        0
    ]
    return builder.make_node("MatMul", [x_flatten, y_flatten], {}, node.name)


def _map_linear(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map linear operation."""
    x = node.inputs.get("X", [])
    y = node.inputs.get("Y", [])
    b = node.inputs.get("Bias", [])
    mm = builder.make_node("MatMul", x + y, {}, f"{node.name}_mm")[0]
    if b:
        return builder.make_node("Add", [mm, b[0]], {}, node.name)
    return [mm]


def _map_conv(op_type: str) -> Callable:
    """Executes the  map conv operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        inputs = []
        if "Input" in node.inputs:
            inputs.extend(node.inputs["Input"])
        if "Filter" in node.inputs:
            inputs.extend(node.inputs["Filter"])
        if "Bias" in node.inputs:
            inputs.extend(node.inputs["Bias"])
        attrs = {
            "strides": builder.extract_list_attr(node, "strides"),
            "pads": builder.extract_list_attr(node, "paddings"),
            "dilations": builder.extract_list_attr(node, "dilations"),
        }
        groups = builder.extract_attr(node, "groups", 1)
        if groups > 1:
            attrs["group"] = groups
        return builder.make_node(op_type, inputs, attrs, node.name)

    return _impl


def _map_pool(op_type: str) -> Callable:
    """Executes the  map pool operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        pooling_type = builder.extract_attr(node, "pooling_type", "max")
        global_pooling = builder.extract_attr(node, "global_pooling", False)
        inputs = node.inputs.get("X", [])
        if global_pooling:
            onnx_op = "GlobalMaxPool" if pooling_type == "max" else "GlobalAveragePool"
            return builder.make_node(onnx_op, inputs, {}, node.name)
        else:
            onnx_op = "MaxPool" if pooling_type == "max" else "AveragePool"
            attrs = {
                "kernel_shape": builder.extract_list_attr(node, "ksize"),
                "strides": builder.extract_list_attr(node, "strides"),
                "pads": builder.extract_list_attr(node, "paddings"),
            }
            return builder.make_node(onnx_op, inputs, attrs, node.name)

    return _impl


def _map_adaptive_pool(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map adaptive pool operation."""
    pooling_type = builder.extract_attr(node, "pooling_type", "max")
    pool_size = builder.extract_list_attr(node, "pool_size")
    inputs = node.inputs.get("X", [])
    if pool_size == [1, 1] or pool_size == [1, 1, 1] or (not pool_size):
        onnx_op = "GlobalMaxPool" if pooling_type == "max" else "GlobalAveragePool"
        return builder.make_node(onnx_op, inputs, {}, node.name)
    else:
        onnx_op = "MaxPool" if pooling_type == "max" else "AveragePool"
        return builder.make_node(onnx_op, inputs, {"kernel_shape": pool_size}, node.name)


def _map_unpool(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map unpool operation."""
    return builder.make_node("MaxUnpool", node.inputs.get("X", []), {}, node.name)


def _map_batch_norm(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map batch norm operation."""
    inputs = []
    for k in ["X", "Scale", "Bias", "Mean", "Variance"]:
        if k in node.inputs:
            inputs.extend(node.inputs[k])
    epsilon = builder.extract_attr(node, "epsilon", 1e-05)
    momentum = builder.extract_attr(node, "momentum", 0.9)
    return builder.make_node(
        "BatchNormalization", inputs, {"epsilon": epsilon, "momentum": momentum}, node.name
    )


def _map_layer_norm(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map layer norm operation."""
    inputs = node.inputs.get("X", [])
    if "Scale" in node.inputs:
        inputs.extend(node.inputs["Scale"])
    if "Bias" in node.inputs:
        inputs.extend(node.inputs["Bias"])
    epsilon = builder.extract_attr(node, "epsilon", 1e-05)
    return builder.make_node("LayerNormalization", inputs, {"epsilon": epsilon}, node.name)


def _map_group_norm(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map group norm operation."""
    inputs = node.inputs.get("X", [])
    x = inputs[0]
    groups = builder.extract_attr(node, "groups", 1)
    epsilon = builder.extract_attr(node, "epsilon", 1e-05)
    shape_node = builder.make_node("Shape", [x], {}, f"{node.name}_shape")[0]
    batch = builder.make_node(
        "Gather",
        [shape_node, builder.add_constant(f"{node.name}_bidx", [0], 7, [1])],
        {"axis": 0},
        f"{node.name}_b",
    )[0]
    c = builder.make_node(
        "Gather",
        [shape_node, builder.add_constant(f"{node.name}_cidx", [1], 7, [1])],
        {"axis": 0},
        f"{node.name}_c",
    )[0]
    h = builder.make_node(
        "Gather",
        [shape_node, builder.add_constant(f"{node.name}_hidx", [2], 7, [1])],
        {"axis": 0},
        f"{node.name}_h",
    )[0]
    w = builder.make_node(
        "Gather",
        [shape_node, builder.add_constant(f"{node.name}_widx", [3], 7, [1])],
        {"axis": 0},
        f"{node.name}_w",
    )[0]
    groups_t = builder.add_constant(f"{node.name}_groups", [groups], 7, [1])
    c_per_g = builder.make_node("Div", [c, groups_t], {}, f"{node.name}_c_per_g")[0]
    n_times_g = builder.make_node("Mul", [batch, groups_t], {}, f"{node.name}_n_g")[0]
    reshape_in_shape = builder.make_node(
        "Concat", [n_times_g, c_per_g, h, w], {"axis": 0}, f"{node.name}_in_shape"
    )[0]
    reshaped_in = builder.make_node(
        "Reshape", [x, reshape_in_shape], {}, f"{node.name}_reshape_in"
    )[0]
    inst_norm = builder.make_node(
        "InstanceNormalization", [reshaped_in], {"epsilon": epsilon}, f"{node.name}_inst"
    )[0]
    reshape_out_shape = builder.make_node(
        "Concat", [batch, c, h, w], {"axis": 0}, f"{node.name}_out_shape"
    )[0]
    return builder.make_node("Reshape", [inst_norm, reshape_out_shape], {}, node.name)


def _map_instance_norm(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map instance norm operation."""
    inputs = node.inputs.get("X", [])
    if "Scale" in node.inputs:
        inputs.extend(node.inputs["Scale"])
    if "Bias" in node.inputs:
        inputs.extend(node.inputs["Bias"])
    epsilon = builder.extract_attr(node, "epsilon", 1e-05)
    return builder.make_node("InstanceNormalization", inputs, {"epsilon": epsilon}, node.name)


def _map_simple_unary(op_type: str) -> Callable:
    """Executes the  map simple unary operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node(op_type, node.inputs.get("X", []), {}, node.name)

    return _impl


def _map_relu6(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map relu6 operation."""
    inputs = node.inputs.get("X", [])
    zero = builder.add_constant(f"{node.name}_zero", 0.0, 1, ())
    six = builder.add_constant(f"{node.name}_six", 6.0, 1, ())
    return builder.make_node("Clip", inputs + [zero, six], {}, node.name)


def _map_leaky_relu(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map leaky relu operation."""
    alpha = builder.extract_attr(node, "alpha", 0.02)
    return builder.make_node("LeakyRelu", node.inputs.get("X", []), {"alpha": alpha}, node.name)


def _map_elu(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map elu operation."""
    alpha = builder.extract_attr(node, "alpha", 1.0)
    return builder.make_node("Elu", node.inputs.get("X", []), {"alpha": alpha}, node.name)


def _map_selu(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map selu operation."""
    alpha = builder.extract_attr(node, "alpha", 1.67326)
    gamma = builder.extract_attr(node, "scale", 1.0507)
    return builder.make_node(
        "Selu", node.inputs.get("X", []), {"alpha": alpha, "gamma": gamma}, node.name
    )


def _map_gelu(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map gelu operation."""
    approximate = builder.extract_attr(node, "approximate", False)
    x = node.inputs.get("X", [])[0]
    if approximate:
        const_0_5 = builder.add_constant(f"{node.name}_0_5", 0.5, 1, [])
        const_sqrt_2_pi = builder.add_constant(f"{node.name}_sqrt_2_pi", 0.7978845608, 1, [])
        const_0_044715 = builder.add_constant(f"{node.name}_0_044715", 0.044715, 1, [])
        const_1 = builder.add_constant(f"{node.name}_1", 1.0, 1, [])
        const_3 = builder.add_constant(f"{node.name}_3", 3.0, 1, [])
        x_cubed = builder.make_node("Pow", [x, const_3], {}, f"{node.name}_x_cubed")[0]
        term1 = builder.make_node("Mul", [x_cubed, const_0_044715], {}, f"{node.name}_term1")[0]
        term2 = builder.make_node("Add", [x, term1], {}, f"{node.name}_term2")[0]
        term3 = builder.make_node("Mul", [term2, const_sqrt_2_pi], {}, f"{node.name}_term3")[0]
        tanh_out = builder.make_node("Tanh", [term3], {}, f"{node.name}_tanh")[0]
        term4 = builder.make_node("Add", [tanh_out, const_1], {}, f"{node.name}_term4")[0]
        term5 = builder.make_node("Mul", [x, term4], {}, f"{node.name}_term5")[0]
        return builder.make_node("Mul", [term5, const_0_5], {}, node.name)
    else:
        const_0_5 = builder.add_constant(f"{node.name}_0_5", 0.5, 1, [])
        const_1 = builder.add_constant(f"{node.name}_1", 1.0, 1, [])
        const_sqrt_2 = builder.add_constant(f"{node.name}_sqrt_2", 1.41421356237, 1, [])
        div_out = builder.make_node("Div", [x, const_sqrt_2], {}, f"{node.name}_div")[0]
        erf_out = builder.make_node("Erf", [div_out], {}, f"{node.name}_erf")[0]
        add_out = builder.make_node("Add", [erf_out, const_1], {}, f"{node.name}_add")[0]
        mul1 = builder.make_node("Mul", [x, add_out], {}, f"{node.name}_mul1")[0]
        return builder.make_node("Mul", [mul1, const_0_5], {}, node.name)


def _map_silu(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map silu operation."""
    inputs = node.inputs.get("X", [])
    sig = builder.make_node("Sigmoid", inputs, {}, f"{node.name}_sig")[0]
    return builder.make_node("Mul", inputs + [sig], {}, node.name)


def _map_hard_swish(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map hard swish operation."""
    return builder.make_node("HardSwish", node.inputs.get("X", []), {}, node.name)


def _map_hard_sigmoid(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map hard sigmoid operation."""
    return builder.make_node("HardSigmoid", node.inputs.get("X", []), {}, node.name)


def _map_softmax(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map softmax operation."""
    axis = builder.extract_attr(node, "axis", -1)
    return builder.make_node("Softmax", node.inputs.get("X", []), {"axis": axis}, node.name)


def _map_log_softmax(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map log softmax operation."""
    axis = builder.extract_attr(node, "axis", -1)
    return builder.make_node("LogSoftmax", node.inputs.get("X", []), {"axis": axis}, node.name)


def _map_dropout(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map dropout operation."""
    prob = builder.extract_attr(node, "dropout_prob", 0.5)
    return builder.make_node("Dropout", node.inputs.get("X", []), {"prob": prob}, node.name)


def _map_pad(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map pad operation."""
    inputs = node.inputs.get("X", [])
    pads = builder.extract_list_attr(node, "paddings")
    pads_const = builder.add_constant(f"{node.name}_pads", pads, 7, (len(pads),))
    value = builder.extract_attr(node, "pad_value", 0.0)
    val_const = builder.add_constant(f"{node.name}_val", value, 1, ())
    return builder.make_node(
        "Pad", inputs + [pads_const, val_const], {"mode": "constant"}, node.name
    )


def _map_l2_normalize(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map l2 normalize operation."""
    axis = builder.extract_attr(node, "axis", -1)
    return builder.make_node(
        "LpNormalization", node.inputs.get("X", []), {"p": 2, "axis": axis}, node.name
    )


def _map_roi_align(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map roi align operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "ROIs" in node.inputs:
        inputs.extend(node.inputs["ROIs"])
    return builder.make_node("RoiAlign", inputs, {}, node.name)


def _map_roi_pool(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map roi pool operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "ROIs" in node.inputs:
        inputs.extend(node.inputs["ROIs"])
    return builder.make_node("MaxRoiPool", inputs, {}, node.name)


def _map_deformable_conv(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map deformable conv operation."""
    return builder.make_node("DeformConv", node.inputs.get("X", []), {}, node.name)


def _map_custom(op_name: str):

    def _impl(builder, node):
        inputs = node.inputs.get("X", [])
        if "Y" in node.inputs:
            inputs.extend(node.inputs["Y"])
        return builder.make_node(op_name, inputs, {}, node.name)

    return _impl


NN_OPS_MAPPING: dict[str, Callable[[PaddleToONNXGraphBuilder, PaddleNode], list[str]]] = {
    "matmul": _map_matmul,
    "matmul_v2": _map_matmul,
    "bmm": _map_matmul,
    "mul": _map_mul,
    "fc": _map_linear,
    "linear": _map_linear,
    "conv2d": _map_conv("Conv"),
    "conv3d": _map_conv("Conv"),
    "depthwise_conv2d": _map_conv("Conv"),
    "conv2d_transpose": _map_conv("ConvTranspose"),
    "conv3d_transpose": _map_conv("ConvTranspose"),
    "pool2d": _map_pool("pool2d"),
    "pool3d": _map_pool("pool3d"),
    "adaptive_pool2d": _map_adaptive_pool,
    "adaptive_pool3d": _map_adaptive_pool,
    "unpool": _map_unpool,
    "batch_norm": _map_batch_norm,
    "sync_batch_norm": _map_batch_norm,
    "layer_norm": _map_layer_norm,
    "group_norm": _map_group_norm,
    "instance_norm": _map_instance_norm,
    "relu": _map_simple_unary("Relu"),
    "relu6": _map_relu6,
    "leaky_relu": _map_leaky_relu,
    "elu": _map_elu,
    "selu": _map_selu,
    "gelu": _map_gelu,
    "silu": _map_silu,
    "swish": _map_silu,
    "hard_swish": _map_hard_swish,
    "hard_sigmoid": _map_hard_sigmoid,
    "softplus": _map_simple_unary("Softplus"),
    "softsign": _map_simple_unary("Softsign"),
    "sigmoid": _map_simple_unary("Sigmoid"),
    "softmax": _map_softmax,
    "log_softmax": _map_log_softmax,
    "dropout": _map_dropout,
    "dropout_nd": _map_dropout,
    "pad": _map_pad,
    "pad2d": _map_pad,
    "pad3d": _map_pad,
    "p_norm": _map_l2_normalize,
    "l2_normalize": _map_l2_normalize,
    "roi_align": _map_roi_align,
    "roi_pool": _map_roi_pool,
    "psroi_pool": _map_roi_pool,
    "deformable_conv": _map_deformable_conv,
    "hard_shrink": _map_custom("HardShrink"),
    "soft_shrink": _map_custom("SoftShrink"),
    "logsigmoid": _map_custom("Custom_Paddle_logsigmoid"),
    "mish": _map_custom("Mish"),
    "prelu": _map_custom("PRelu"),
    "bipolar_sigmoid": _map_custom("Custom_Paddle_bipolar_sigmoid"),
    "max_pool2d_with_index": _map_custom("MaxPool"),
    "bipartite_match": _map_custom("Custom_Paddle_bipartite_match"),
    "affine_channel": _map_custom("Custom_Paddle_affine_channel"),
    "anchor_generator": _map_custom("Custom_Paddle_anchor_generator"),
    "collect_fpn_proposals": _map_custom("Custom_Paddle_collect_fpn_proposals"),
    "deformable_conv_v1": _map_custom("Custom_Paddle_deformable_conv_v1"),
    "multihead_attention": _map_custom("Custom_Paddle_multihead_attention"),
    "bce_loss": _map_custom("Custom_Paddle_bce_loss"),
    "cross_entropy": _map_custom("Custom_Paddle_cross_entropy"),
    "huber_loss": _map_custom("Custom_Paddle_huber_loss"),
    "l1_loss": _map_custom("Custom_Paddle_l1_loss"),
    "mse_loss": _map_custom("Custom_Paddle_mse_loss"),
    "nll_loss": _map_custom("NegativeLogLikelihoodLoss"),
    "smooth_l1_loss": _map_custom("Custom_Paddle_smooth_l1_loss"),
    "sigmoid_cross_entropy_with_logits": _map_custom(
        "Custom_Paddle_sigmoid_cross_entropy_with_logits"
    ),
    "softmax_with_cross_entropy": _map_custom("SoftmaxCrossEntropyLoss"),
}
