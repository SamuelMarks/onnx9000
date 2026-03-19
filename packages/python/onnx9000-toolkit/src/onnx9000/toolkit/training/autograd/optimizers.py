"""Optimizer Generators.

Constructs pure-ONNX subgraphs for various weight update algorithms (SGD, Adam, etc.)
"""

from onnx9000.core.ir import Graph, Node


def add_sgd_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    weight_decay: float = 0.0,
    momentum: float = 0.0,
) -> None:
    """Add SGD optimizer steps to the graph."""
    for param in parameters:
        grad = f"grad_{param}"
        if weight_decay > 0.0:
            wd_grad = f"{grad}_wd"
            graph.add_node(
                Node(
                    "ConstantOfShape",
                    [f"shape_{param}"],
                    [f"wd_const_{param}"],
                    {"value": weight_decay},
                    name=f"{param}_wd_const",
                )
            )
            graph.add_node(
                Node(
                    "Mul",
                    [param, f"wd_const_{param}"],
                    [f"{param}_wd_term"],
                    {},
                    name=f"{param}_wd_mul",
                )
            )
            graph.add_node(
                Node("Add", [grad, f"{param}_wd_term"], [wd_grad], {}, name=f"{param}_wd_add")
            )
            grad = wd_grad
        update = f"{param}_update"
        if momentum > 0.0:
            v = f"momentum_{param}"
            v_new = f"momentum_{param}_new"
            graph.add_node(
                Node(
                    "ConstantOfShape",
                    [f"shape_{param}"],
                    [f"mom_const_{param}"],
                    {"value": momentum},
                    name=f"{param}_mom_const",
                )
            )
            graph.add_node(
                Node(
                    "Mul",
                    [v, f"mom_const_{param}"],
                    [f"{param}_mom_term"],
                    {},
                    name=f"{param}_mom_mul",
                )
            )
            graph.add_node(
                Node("Add", [f"{param}_mom_term", grad], [v_new], {}, name=f"{param}_mom_add")
            )
            graph.add_node(
                Node("Mul", [v_new, learning_rate], [update], {}, name=f"{param}_lr_mul_mom")
            )
        else:
            graph.add_node(Node("Mul", [grad, learning_rate], [update], {}, name=f"{param}_lr_mul"))
        graph.add_node(
            Node("Sub", [param, update], [f"{param}_new"], {}, name=f"{param}_update_sub")
        )


def add_adam_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-08,
    weight_decay: float = 0.0,
    global_step: str = "global_step",
) -> None:
    """Add Adam optimizer steps natively to the graph."""
    if global_step not in graph.inputs:
        graph.inputs.append(global_step)

    # Precompute bias correction terms globally
    graph.add_node(Node("Constant", [], ["_adam_b1"], {"value": [beta1]}, name="adam_b1_const"))
    graph.add_node(Node("Constant", [], ["_adam_b2"], {"value": [beta2]}, name="adam_b2_const"))
    graph.add_node(Node("Constant", [], ["_adam_eps"], {"value": [epsilon]}, name="adam_eps_const"))
    graph.add_node(Node("Constant", [], ["_adam_1"], {"value": [1.0]}, name="adam_1_const"))

    graph.add_node(Node("Sub", ["_adam_1", "_adam_b1"], ["_adam_1_m_b1"], {}, name="adam_sub_b1"))
    graph.add_node(Node("Sub", ["_adam_1", "_adam_b2"], ["_adam_1_m_b2"], {}, name="adam_sub_b2"))

    # Bias correction components
    graph.add_node(Node("Pow", ["_adam_b1", global_step], ["_adam_b1_t"], {}, name="adam_pow_b1"))
    graph.add_node(Node("Pow", ["_adam_b2", global_step], ["_adam_b2_t"], {}, name="adam_pow_b2"))
    graph.add_node(Node("Sub", ["_adam_1", "_adam_b1_t"], ["_adam_bias_m"], {}, name="adam_bias_m"))
    graph.add_node(Node("Sub", ["_adam_1", "_adam_b2_t"], ["_adam_bias_v"], {}, name="adam_bias_v"))

    for param in parameters:
        grad = f"grad_{param}"
        m = f"adam_m_{param}"
        v = f"adam_v_{param}"

        m_new = f"{m}_new"
        v_new = f"{v}_new"
        param_new = f"{param}_new"

        shape_name = f"shape_{param}"
        if shape_name not in graph.inputs and not any(
            n.outputs[0] == shape_name for n in graph.nodes
        ):
            graph.add_node(Node("Shape", [param], [shape_name], {}, name=f"{param}_adam_shape"))

        if m not in graph.inputs and m not in graph.initializers:
            # Auto-generate initialization values for optimizer states (Zero tensors) dynamically
            graph.add_node(
                Node(
                    "ConstantOfShape",
                    [shape_name],
                    [m],
                    {"value": 0.0},
                    name=f"{param}_adam_init_m",
                )
            )

        if v not in graph.inputs and v not in graph.initializers:
            graph.add_node(
                Node(
                    "ConstantOfShape",
                    [shape_name],
                    [v],
                    {"value": 0.0},
                    name=f"{param}_adam_init_v",
                )
            )
        graph.outputs.extend([m_new, v_new])

        if weight_decay > 0.0:
            wd_grad = f"{grad}_wd"
            graph.add_node(
                Node(
                    "ConstantOfShape",
                    [f"shape_{param}"],
                    [f"wd_const_{param}"],
                    {"value": weight_decay},
                    name=f"{param}_wd_c",
                )
            )
            graph.add_node(
                Node(
                    "Mul",
                    [param, f"wd_const_{param}"],
                    [f"{param}_wd_term"],
                    {},
                    name=f"{param}_wd_mul",
                )
            )
            graph.add_node(
                Node("Add", [grad, f"{param}_wd_term"], [wd_grad], {}, name=f"{param}_wd_add")
            )
            grad = wd_grad

        # Update biased first moment estimate: m_new = beta1 * m + (1 - beta1) * grad
        m_term1 = f"{param}_m_t1"
        m_term2 = f"{param}_m_t2"
        graph.add_node(Node("Mul", [m, "_adam_b1"], [m_term1], {}, name=f"{param}_mul_m_b1"))
        graph.add_node(Node("Mul", [grad, "_adam_1_m_b1"], [m_term2], {}, name=f"{param}_mul_g_b1"))
        graph.add_node(Node("Add", [m_term1, m_term2], [m_new], {}, name=f"{param}_add_m"))

        # Update biased second raw moment estimate: v_new = beta2 * v + (1 - beta2) * grad^2
        grad_sq = f"{param}_g_sq"
        v_term1 = f"{param}_v_t1"
        v_term2 = f"{param}_v_t2"
        graph.add_node(Node("Mul", [grad, grad], [grad_sq], {}, name=f"{param}_mul_g_sq"))
        graph.add_node(Node("Mul", [v, "_adam_b2"], [v_term1], {}, name=f"{param}_mul_v_b2"))
        graph.add_node(
            Node("Mul", [grad_sq, "_adam_1_m_b2"], [v_term2], {}, name=f"{param}_mul_g_sq_b2")
        )
        graph.add_node(Node("Add", [v_term1, v_term2], [v_new], {}, name=f"{param}_add_v"))

        # Compute bias-corrected first moment estimate
        m_hat = f"{param}_m_hat"
        graph.add_node(Node("Div", [m_new, "_adam_bias_m"], [m_hat], {}, name=f"{param}_div_m_hat"))

        # Compute bias-corrected second raw moment estimate
        v_hat = f"{param}_v_hat"
        graph.add_node(Node("Div", [v_new, "_adam_bias_v"], [v_hat], {}, name=f"{param}_div_v_hat"))

        # Compute update
        v_hat_sqrt = f"{param}_v_hat_sqrt"
        denom = f"{param}_denom"
        update_step = f"{param}_update_step"
        lr_scaled = f"{param}_lr_scaled"

        graph.add_node(Node("Sqrt", [v_hat], [v_hat_sqrt], {}, name=f"{param}_sqrt_v"))
        graph.add_node(Node("Add", [v_hat_sqrt, "_adam_eps"], [denom], {}, name=f"{param}_add_eps"))
        graph.add_node(Node("Div", [m_hat, denom], [update_step], {}, name=f"{param}_div_upd"))
        graph.add_node(
            Node("Mul", [update_step, learning_rate], [lr_scaled], {}, name=f"{param}_mul_lr")
        )

        # Apply update
        graph.add_node(Node("Sub", [param, lr_scaled], [param_new], {}, name=f"{param}_sub_param"))


def add_adamw_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-08,
    weight_decay: float = 0.01,
    global_step: str = "global_step",
) -> None:
    """Add AdamW optimizer steps natively to the graph."""
    if global_step not in graph.inputs:
        graph.inputs.append(global_step)

    graph.add_node(Node("Constant", [], ["_adamw_b1"], {"value": [beta1]}, name="adamw_b1_const"))
    graph.add_node(Node("Constant", [], ["_adamw_b2"], {"value": [beta2]}, name="adamw_b2_const"))
    graph.add_node(
        Node("Constant", [], ["_adamw_eps"], {"value": [epsilon]}, name="adamw_eps_const")
    )
    graph.add_node(Node("Constant", [], ["_adamw_1"], {"value": [1.0]}, name="adamw_1_const"))
    graph.add_node(
        Node("Constant", [], ["_adamw_wd"], {"value": [weight_decay]}, name="adamw_wd_const")
    )

    graph.add_node(
        Node("Sub", ["_adamw_1", "_adamw_b1"], ["_adamw_1_m_b1"], {}, name="adamw_sub_b1")
    )
    graph.add_node(
        Node("Sub", ["_adamw_1", "_adamw_b2"], ["_adamw_1_m_b2"], {}, name="adamw_sub_b2")
    )

    graph.add_node(
        Node("Pow", ["_adamw_b1", global_step], ["_adamw_b1_t"], {}, name="adamw_pow_b1")
    )
    graph.add_node(
        Node("Pow", ["_adamw_b2", global_step], ["_adamw_b2_t"], {}, name="adamw_pow_b2")
    )
    graph.add_node(
        Node("Sub", ["_adamw_1", "_adamw_b1_t"], ["_adamw_bias_m"], {}, name="adamw_bias_m")
    )
    graph.add_node(
        Node("Sub", ["_adamw_1", "_adamw_b2_t"], ["_adamw_bias_v"], {}, name="adamw_bias_v")
    )

    # Compute wd_scaled = learning_rate * weight_decay
    graph.add_node(
        Node("Mul", [learning_rate, "_adamw_wd"], ["_adamw_lr_wd"], {}, name="adamw_lr_wd_mul")
    )

    for param in parameters:
        grad = f"grad_{param}"
        m = f"adamw_m_{param}"
        v = f"adamw_v_{param}"

        m_new = f"{m}_new"
        v_new = f"{v}_new"
        param_new = f"{param}_new"

        if m not in graph.inputs:
            graph.inputs.append(m)
        if v not in graph.inputs:
            graph.inputs.append(v)

        graph.outputs.extend([m_new, v_new])

        # In AdamW, weight decay is applied directly to the parameter, decoupled from gradient:
        # param = param - lr * weight_decay * param
        param_wd_update = f"{param}_wd_upd"
        param_decayed = f"{param}_decayed"
        graph.add_node(
            Node("Mul", [param, "_adamw_lr_wd"], [param_wd_update], {}, name=f"{param}_mul_wd")
        )
        graph.add_node(
            Node("Sub", [param, param_wd_update], [param_decayed], {}, name=f"{param}_sub_wd")
        )

        # Update biased first moment
        m_term1 = f"{param}_m_t1"
        m_term2 = f"{param}_m_t2"
        graph.add_node(Node("Mul", [m, "_adamw_b1"], [m_term1], {}, name=f"{param}_mul_m_b1"))
        graph.add_node(
            Node("Mul", [grad, "_adamw_1_m_b1"], [m_term2], {}, name=f"{param}_mul_g_b1")
        )
        graph.add_node(Node("Add", [m_term1, m_term2], [m_new], {}, name=f"{param}_add_m"))

        # Update biased second raw moment
        grad_sq = f"{param}_g_sq"
        v_term1 = f"{param}_v_t1"
        v_term2 = f"{param}_v_t2"
        graph.add_node(Node("Mul", [grad, grad], [grad_sq], {}, name=f"{param}_mul_g_sq"))
        graph.add_node(Node("Mul", [v, "_adamw_b2"], [v_term1], {}, name=f"{param}_mul_v_b2"))
        graph.add_node(
            Node("Mul", [grad_sq, "_adamw_1_m_b2"], [v_term2], {}, name=f"{param}_mul_g_sq_b2")
        )
        graph.add_node(Node("Add", [v_term1, v_term2], [v_new], {}, name=f"{param}_add_v"))

        # Compute bias-corrected
        m_hat = f"{param}_m_hat"
        graph.add_node(
            Node("Div", [m_new, "_adamw_bias_m"], [m_hat], {}, name=f"{param}_div_m_hat")
        )
        v_hat = f"{param}_v_hat"
        graph.add_node(
            Node("Div", [v_new, "_adamw_bias_v"], [v_hat], {}, name=f"{param}_div_v_hat")
        )

        # Compute step
        v_hat_sqrt = f"{param}_v_hat_sqrt"
        denom = f"{param}_denom"
        update_step = f"{param}_update_step"
        lr_scaled = f"{param}_lr_scaled"

        graph.add_node(Node("Sqrt", [v_hat], [v_hat_sqrt], {}, name=f"{param}_sqrt_v"))
        graph.add_node(
            Node("Add", [v_hat_sqrt, "_adamw_eps"], [denom], {}, name=f"{param}_add_eps")
        )
        graph.add_node(Node("Div", [m_hat, denom], [update_step], {}, name=f"{param}_div_upd"))
        graph.add_node(
            Node("Mul", [update_step, learning_rate], [lr_scaled], {}, name=f"{param}_mul_lr")
        )

        # Apply update to the decayed parameter
        graph.add_node(
            Node("Sub", [param_decayed, lr_scaled], [param_new], {}, name=f"{param}_sub_param")
        )


def add_rmsprop_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    alpha: float = 0.99,
    epsilon: float = 1e-08,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
) -> None:
    """Add RMSprop optimizer steps natively to the graph."""
    graph.add_node(
        Node("Constant", [], ["_rmsprop_alpha"], {"value": [alpha]}, name="rmsprop_alpha_c")
    )
    graph.add_node(
        Node("Constant", [], ["_rmsprop_1_m_alpha"], {"value": [1.0 - alpha]}, name="rmsprop_1ma_c")
    )
    graph.add_node(
        Node("Constant", [], ["_rmsprop_eps"], {"value": [epsilon]}, name="rmsprop_eps_c")
    )

    if momentum > 0.0:
        graph.add_node(
            Node("Constant", [], ["_rmsprop_mom"], {"value": [momentum]}, name="rmsprop_mom_c")
        )

    for param in parameters:
        grad = f"grad_{param}"
        v = f"rmsprop_v_{param}"
        v_new = f"{v}_new"
        param_new = f"{param}_new"

        if v not in graph.inputs:
            graph.inputs.append(v)
        graph.outputs.append(v_new)

        if momentum > 0.0:
            m = f"rmsprop_m_{param}"
            m_new = f"{m}_new"
            if m not in graph.inputs:
                graph.inputs.append(m)
            graph.outputs.append(m_new)

        if weight_decay > 0.0:
            wd_grad = f"{grad}_wd"
            graph.add_node(
                Node(
                    "ConstantOfShape",
                    [f"shape_{param}"],
                    [f"wd_const_{param}"],
                    {"value": weight_decay},
                    name=f"{param}_wd_c",
                )
            )
            graph.add_node(
                Node(
                    "Mul",
                    [param, f"wd_const_{param}"],
                    [f"{param}_wd_term"],
                    {},
                    name=f"{param}_wd_mul",
                )
            )
            graph.add_node(
                Node("Add", [grad, f"{param}_wd_term"], [wd_grad], {}, name=f"{param}_wd_add")
            )
            grad = wd_grad

        # Update moving average of squared gradients: v_new = alpha * v + (1 - alpha) * grad^2
        grad_sq = f"{param}_g_sq"
        v_term1 = f"{param}_v_t1"
        v_term2 = f"{param}_v_t2"
        graph.add_node(Node("Mul", [grad, grad], [grad_sq], {}, name=f"{param}_mul_g_sq"))
        graph.add_node(
            Node("Mul", [v, "_rmsprop_alpha"], [v_term1], {}, name=f"{param}_mul_v_alpha")
        )
        graph.add_node(
            Node(
                "Mul", [grad_sq, "_rmsprop_1_m_alpha"], [v_term2], {}, name=f"{param}_mul_g_sq_1ma"
            )
        )
        graph.add_node(Node("Add", [v_term1, v_term2], [v_new], {}, name=f"{param}_add_v"))

        # Calculate step
        v_sqrt = f"{param}_v_sqrt"
        denom = f"{param}_denom"
        update_step = f"{param}_update_step"

        graph.add_node(Node("Sqrt", [v_new], [v_sqrt], {}, name=f"{param}_sqrt_v"))
        graph.add_node(Node("Add", [v_sqrt, "_rmsprop_eps"], [denom], {}, name=f"{param}_add_eps"))
        graph.add_node(Node("Div", [grad, denom], [update_step], {}, name=f"{param}_div_upd"))

        if momentum > 0.0:
            # m_new = momentum * m + update_step
            # param_new = param - lr * m_new
            m_term1 = f"{param}_m_t1"
            graph.add_node(
                Node("Mul", [m, "_rmsprop_mom"], [m_term1], {}, name=f"{param}_mul_m_mom")
            )
            graph.add_node(Node("Add", [m_term1, update_step], [m_new], {}, name=f"{param}_add_m"))

            lr_scaled = f"{param}_lr_scaled"
            graph.add_node(
                Node("Mul", [m_new, learning_rate], [lr_scaled], {}, name=f"{param}_mul_lr_m")
            )
            graph.add_node(
                Node("Sub", [param, lr_scaled], [param_new], {}, name=f"{param}_sub_param")
            )
        else:
            # param_new = param - lr * update_step
            lr_scaled = f"{param}_lr_scaled"
            graph.add_node(
                Node("Mul", [update_step, learning_rate], [lr_scaled], {}, name=f"{param}_mul_lr")
            )
            graph.add_node(
                Node("Sub", [param, lr_scaled], [param_new], {}, name=f"{param}_sub_param")
            )


def add_adagrad_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    epsilon: float = 1e-10,
    weight_decay: float = 0.0,
) -> None:
    """Add Adagrad optimizer steps natively to the graph."""
    graph.add_node(
        Node("Constant", [], ["_adagrad_eps"], {"value": [epsilon]}, name="adagrad_eps_c")
    )

    for param in parameters:
        grad = f"grad_{param}"
        v = f"adagrad_v_{param}"
        v_new = f"{v}_new"
        param_new = f"{param}_new"

        if v not in graph.inputs:
            graph.inputs.append(v)
        graph.outputs.append(v_new)

        if weight_decay > 0.0:
            wd_grad = f"{grad}_wd"
            graph.add_node(
                Node(
                    "ConstantOfShape",
                    [f"shape_{param}"],
                    [f"wd_const_{param}"],
                    {"value": weight_decay},
                    name=f"{param}_wd_c",
                )
            )
            graph.add_node(
                Node(
                    "Mul",
                    [param, f"wd_const_{param}"],
                    [f"{param}_wd_term"],
                    {},
                    name=f"{param}_wd_mul",
                )
            )
            graph.add_node(
                Node("Add", [grad, f"{param}_wd_term"], [wd_grad], {}, name=f"{param}_wd_add")
            )
            grad = wd_grad

        # Update sum of squared gradients: v_new = v + grad^2
        grad_sq = f"{param}_g_sq"
        graph.add_node(Node("Mul", [grad, grad], [grad_sq], {}, name=f"{param}_mul_g_sq"))
        graph.add_node(Node("Add", [v, grad_sq], [v_new], {}, name=f"{param}_add_v"))

        # Compute step: update = lr * grad / (sqrt(v_new) + eps)
        v_sqrt = f"{param}_v_sqrt"
        denom = f"{param}_denom"
        update_step = f"{param}_update_step"
        lr_scaled = f"{param}_lr_scaled"

        graph.add_node(Node("Sqrt", [v_new], [v_sqrt], {}, name=f"{param}_sqrt_v"))
        graph.add_node(Node("Add", [v_sqrt, "_adagrad_eps"], [denom], {}, name=f"{param}_add_eps"))
        graph.add_node(Node("Div", [grad, denom], [update_step], {}, name=f"{param}_div_upd"))
        graph.add_node(
            Node("Mul", [update_step, learning_rate], [lr_scaled], {}, name=f"{param}_mul_lr")
        )

        # Apply update
        graph.add_node(Node("Sub", [param, lr_scaled], [param_new], {}, name=f"{param}_sub_param"))


def add_adadelta_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    rho: float = 0.9,
    epsilon: float = 1e-06,
    weight_decay: float = 0.0,
) -> None:
    """Add Adadelta optimizer steps natively to the graph."""
    graph.add_node(Node("Constant", [], ["_adadelta_rho"], {"value": [rho]}, name="adadelta_rho_c"))
    graph.add_node(
        Node("Constant", [], ["_adadelta_1_m_rho"], {"value": [1.0 - rho]}, name="adadelta_1mrho_c")
    )
    graph.add_node(
        Node("Constant", [], ["_adadelta_eps"], {"value": [epsilon]}, name="adadelta_eps_c")
    )

    for param in parameters:
        grad = f"grad_{param}"
        v = f"adadelta_v_{param}"
        u = f"adadelta_u_{param}"
        v_new = f"{v}_new"
        u_new = f"{u}_new"
        param_new = f"{param}_new"

        if v not in graph.inputs:
            graph.inputs.append(v)
        if u not in graph.inputs:
            graph.inputs.append(u)

        graph.outputs.extend([v_new, u_new])

        if weight_decay > 0.0:
            wd_grad = f"{grad}_wd"
            graph.add_node(
                Node(
                    "ConstantOfShape",
                    [f"shape_{param}"],
                    [f"wd_const_{param}"],
                    {"value": weight_decay},
                    name=f"{param}_wd_c",
                )
            )
            graph.add_node(
                Node(
                    "Mul",
                    [param, f"wd_const_{param}"],
                    [f"{param}_wd_term"],
                    {},
                    name=f"{param}_wd_mul",
                )
            )
            graph.add_node(
                Node("Add", [grad, f"{param}_wd_term"], [wd_grad], {}, name=f"{param}_wd_add")
            )
            grad = wd_grad

        # Update moving average of squared gradients: v_new = rho * v + (1 - rho) * grad^2
        grad_sq = f"{param}_g_sq"
        v_term1 = f"{param}_v_t1"
        v_term2 = f"{param}_v_t2"
        graph.add_node(Node("Mul", [grad, grad], [grad_sq], {}, name=f"{param}_mul_g_sq"))
        graph.add_node(Node("Mul", [v, "_adadelta_rho"], [v_term1], {}, name=f"{param}_mul_v_rho"))
        graph.add_node(
            Node(
                "Mul", [grad_sq, "_adadelta_1_m_rho"], [v_term2], {}, name=f"{param}_mul_g_sq_1mrho"
            )
        )
        graph.add_node(Node("Add", [v_term1, v_term2], [v_new], {}, name=f"{param}_add_v"))

        # Compute update step: delta_x = sqrt(u + eps) / sqrt(v_new + eps) * grad
        u_sqrt = f"{param}_u_sqrt"
        v_sqrt = f"{param}_v_sqrt"
        u_add_eps = f"{param}_u_add_eps"
        v_add_eps = f"{param}_v_add_eps"
        ratio = f"{param}_ratio"
        update_step = f"{param}_update_step"

        graph.add_node(
            Node("Add", [u, "_adadelta_eps"], [u_add_eps], {}, name=f"{param}_u_add_eps")
        )
        graph.add_node(Node("Sqrt", [u_add_eps], [u_sqrt], {}, name=f"{param}_u_sqrt"))
        graph.add_node(
            Node("Add", [v_new, "_adadelta_eps"], [v_add_eps], {}, name=f"{param}_v_add_eps")
        )
        graph.add_node(Node("Sqrt", [v_add_eps], [v_sqrt], {}, name=f"{param}_v_sqrt"))
        graph.add_node(Node("Div", [u_sqrt, v_sqrt], [ratio], {}, name=f"{param}_div_ratio"))
        graph.add_node(Node("Mul", [ratio, grad], [update_step], {}, name=f"{param}_mul_upd"))

        # Update moving average of squared updates: u_new = rho * u + (1 - rho) * update_step^2
        upd_sq = f"{param}_upd_sq"
        u_term1 = f"{param}_u_t1"
        u_term2 = f"{param}_u_t2"
        graph.add_node(
            Node("Mul", [update_step, update_step], [upd_sq], {}, name=f"{param}_mul_upd_sq")
        )
        graph.add_node(Node("Mul", [u, "_adadelta_rho"], [u_term1], {}, name=f"{param}_mul_u_rho"))
        graph.add_node(
            Node(
                "Mul",
                [upd_sq, "_adadelta_1_m_rho"],
                [u_term2],
                {},
                name=f"{param}_mul_upd_sq_1mrho",
            )
        )
        graph.add_node(Node("Add", [u_term1, u_term2], [u_new], {}, name=f"{param}_add_u"))

        # Apply update
        lr_scaled = f"{param}_lr_scaled"
        graph.add_node(
            Node("Mul", [update_step, learning_rate], [lr_scaled], {}, name=f"{param}_mul_lr")
        )
        graph.add_node(Node("Sub", [param, lr_scaled], [param_new], {}, name=f"{param}_sub_param"))


def add_gradient_accumulation(graph: Graph, grad_names: list[str], steps: int) -> None:
    """Add gradient accumulation logic."""
    return


def add_differential_privacy_noise(
    graph: Graph, grad_names: list[str], noise_multiplier: float, max_grad_norm: float
) -> None:
    """Implement Differential Privacy natively by adding RandomNormal noise to.

    Gradients explicitly before export.
    """
    if not grad_names or noise_multiplier <= 0.0:
        return

    std_dev = noise_multiplier * max_grad_norm
    for i, g in enumerate(grad_names):
        noise_out = f"{g}_dp_noise"
        noisy_grad = f"{g}_dp_noisy"
        shape_in = f"shape_{g}"

        if shape_in not in graph.inputs and not any(n.outputs[0] == shape_in for n in graph.nodes):
            graph.add_node(Node("Shape", [g], [shape_in], {}, name=f"{g}_shape_dp"))

        graph.add_node(
            Node(
                "RandomNormalLike",
                [g],
                [noise_out],
                {"mean": 0.0, "scale": std_dev},
                name=f"{g}_dp_rand",
            )
        )

        graph.add_node(Node("Add", [g, noise_out], [noisy_grad], {}, name=f"{g}_dp_add"))
        grad_names[i] = noisy_grad


def add_gradient_clipping(graph: Graph, grad_names: list[str], max_norm: float) -> None:
    """Add gradient clipping by global norm logic natively."""
    if not grad_names or max_norm <= 0.0:
        return
    sq_norms = []
    for g in grad_names:
        sq_g = f"{g}_sq"
        sum_sq_g = f"{g}_sum_sq"
        graph.add_node(Node("Mul", [g, g], [sq_g], {}, name=f"{g}_clip_mul"))
        graph.add_node(Node("ReduceSum", [sq_g], [sum_sq_g], {"keepdims": 0}, name=f"{g}_clip_sum"))
        sq_norms.append(sum_sq_g)

    global_sq_norm = "global_sq_norm_tmp"
    if len(sq_norms) == 1:
        global_sq_norm = sq_norms[0]
    else:
        graph.add_node(Node("Sum", sq_norms, [global_sq_norm], {}, name="global_sq_norm_sum"))

    global_norm = "global_norm"
    graph.add_node(Node("Sqrt", [global_sq_norm], [global_norm], {}, name="global_norm_sqrt"))

    # clip_scale = min(max_norm / global_norm, 1.0)
    c_max_norm = "clip_max_norm"
    graph.add_node(
        Node("Constant", [], [c_max_norm], {"value": [max_norm]}, name="clip_max_norm_c")
    )
    norm_ratio = "norm_ratio"
    graph.add_node(Node("Div", [c_max_norm, global_norm], [norm_ratio], {}, name="norm_ratio_div"))

    one_const = "one_const"
    graph.add_node(Node("Constant", [], [one_const], {"value": [1.0]}, name="one_const_clip"))

    norm_exceeds = "norm_exceeds"
    graph.add_node(
        Node("Greater", [global_norm, c_max_norm], [norm_exceeds], {}, name="norm_exceeds_cmp")
    )

    graph.add_node(
        Node(
            "Where",
            ["norm_exceeds", "norm_ratio", "one_const"],
            ["clip_scale"],
            {},
            name="norm_where",
        )
    )
    for i, g in enumerate(grad_names):
        clipped_g = f"{g}_clipped"
        graph.add_node(Node("Mul", [g, "clip_scale"], [clipped_g], {}, name=f"{g}_clip_apply"))
        grad_names[i] = clipped_g


def add_local_dp_gradient_clipping(graph: Graph, grad_names: list[str], max_l2_norm: float) -> None:
    """Implement Gradient Clipping to L2 Norm (Local DP constraints) statically.

    Differs from global norm by clipping each gradient independently.
    """
    if not grad_names or max_l2_norm <= 0.0:
        return

    c_max_norm = "local_dp_max_norm"
    graph.add_node(
        Node("Constant", [], [c_max_norm], {"value": [max_l2_norm]}, name="local_dp_c_max_norm")
    )

    for i, g in enumerate(grad_names):
        sq_g = f"{g}_dp_sq"
        sum_sq_g = f"{g}_dp_sum_sq"
        norm_g = f"{g}_dp_norm"
        graph.add_node(Node("Mul", [g, g], [sq_g], {}, name=f"{g}_dp_mul"))
        graph.add_node(Node("ReduceSum", [sq_g], [sum_sq_g], {"keepdims": 0}, name=f"{g}_dp_sum"))
        graph.add_node(Node("Sqrt", [sum_sq_g], [norm_g], {}, name=f"{g}_dp_sqrt"))

        ratio = f"{g}_dp_ratio"
        graph.add_node(Node("Div", [c_max_norm, norm_g], [ratio], {}, name=f"{g}_dp_div"))

        # if norm > max_norm: scale = ratio else 1.0
        c_1 = f"{g}_dp_c1"
        graph.add_node(Node("Constant", [], [c_1], {"value": [1.0]}, name=f"{g}_dp_c1_node"))

        cond = f"{g}_dp_cond"
        scale = f"{g}_dp_scale"
        graph.add_node(Node("Greater", [norm_g, c_max_norm], [cond], {}, name=f"{g}_dp_greater"))
        graph.add_node(Node("Where", [cond, ratio, c_1], [scale], {}, name=f"{g}_dp_where"))

        clipped_g = f"{g}_dp_clipped"
        graph.add_node(Node("Mul", [g, scale], [clipped_g], {}, name=f"{g}_dp_apply"))
        grad_names[i] = clipped_g


def add_gradient_clipping_value(graph: Graph, grad_names: list[str], clip_value: float) -> None:
    """Add gradient clipping by absolute value natively."""
    if not grad_names or clip_value <= 0.0:
        return

    c_min = "clip_val_min"
    c_max = "clip_val_max"
    graph.add_node(Node("Constant", [], [c_min], {"value": [-clip_value]}, name="clip_min_c"))
    graph.add_node(Node("Constant", [], [c_max], {"value": [clip_value]}, name="clip_max_c"))

    for i, g in enumerate(grad_names):
        clipped_g = f"{g}_val_clipped"
        graph.add_node(Node("Clip", [g, c_min, c_max], [clipped_g], {}, name=f"{g}_clip_val_node"))
        grad_names[i] = clipped_g
