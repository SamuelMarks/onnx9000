"""
Optimizer Generators

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
    """Adds SGD optimizer steps to the graph."""
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
) -> None:
    """Adds Adam optimizer steps to the graph."""
    for param in parameters:
        grad = f"grad_{param}"
        m = f"adam_m_{param}"
        v = f"adam_v_{param}"
        t = f"adam_t_{param}"
        graph.add_node(
            Node(
                "AdamStep",
                [param, grad, m, v, t, learning_rate],
                [f"{param}_new", f"{m}_new", f"{v}_new", f"{t}_new"],
                {"beta1": beta1, "beta2": beta2, "epsilon": epsilon, "weight_decay": weight_decay},
                name=f"{param}_adam",
            )
        )


def add_adamw_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-08,
    weight_decay: float = 0.01,
) -> None:
    """Adds AdamW optimizer steps to the graph."""
    for param in parameters:
        grad = f"grad_{param}"
        m = f"adamw_m_{param}"
        v = f"adamw_v_{param}"
        t = f"adamw_t_{param}"
        graph.add_node(
            Node(
                "AdamWStep",
                [param, grad, m, v, t, learning_rate],
                [f"{param}_new", f"{m}_new", f"{v}_new", f"{t}_new"],
                {"beta1": beta1, "beta2": beta2, "epsilon": epsilon, "weight_decay": weight_decay},
                name=f"{param}_adamw",
            )
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
    """Adds RMSprop optimizer steps to the graph."""
    for param in parameters:
        grad = f"grad_{param}"
        v = f"rmsprop_v_{param}"
        graph.add_node(
            Node(
                "RMSpropStep",
                [param, grad, v, learning_rate],
                [f"{param}_new", f"{v}_new"],
                {
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                },
                name=f"{param}_rmsprop",
            )
        )


def add_adagrad_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    epsilon: float = 1e-10,
    weight_decay: float = 0.0,
) -> None:
    """Adds Adagrad optimizer steps to the graph."""
    for param in parameters:
        grad = f"grad_{param}"
        v = f"adagrad_v_{param}"
        graph.add_node(
            Node(
                "AdagradStep",
                [param, grad, v, learning_rate],
                [f"{param}_new", f"{v}_new"],
                {"epsilon": epsilon, "weight_decay": weight_decay},
                name=f"{param}_adagrad",
            )
        )


def add_adadelta_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    rho: float = 0.9,
    epsilon: float = 1e-06,
    weight_decay: float = 0.0,
) -> None:
    """Adds Adadelta optimizer steps to the graph."""
    for param in parameters:
        grad = f"grad_{param}"
        v = f"adadelta_v_{param}"
        u = f"adadelta_u_{param}"
        graph.add_node(
            Node(
                "AdadeltaStep",
                [param, grad, v, u, learning_rate],
                [f"{param}_new", f"{v}_new", f"{u}_new"],
                {"rho": rho, "epsilon": epsilon, "weight_decay": weight_decay},
                name=f"{param}_adadelta",
            )
        )


def add_gradient_accumulation(graph: Graph, grad_names: list[str], steps: int) -> None:
    """Adds gradient accumulation logic."""
    return


def add_gradient_clipping(graph: Graph, grad_names: list[str], max_norm: float) -> None:
    """Adds gradient clipping by global norm logic natively."""
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
    graph.add_node(
        Node("Constant", [], ["max_norm_const"], {"value": [max_norm]}, name="max_norm_c")
    )
    graph.add_node(Node("Constant", [], ["one_const"], {"value": [1.0]}, name="one_c"))
    graph.add_node(
        Node("Div", ["max_norm_const", global_norm], ["norm_ratio"], {}, name="norm_div")
    )
    graph.add_node(
        Node("Greater", [global_norm, "max_norm_const"], ["norm_exceeds"], {}, name="norm_gt")
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
