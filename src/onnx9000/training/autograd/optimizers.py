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
            # grad = grad + weight_decay * param
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
                Node(
                    "Add",
                    [grad, f"{param}_wd_term"],
                    [wd_grad],
                    {},
                    name=f"{param}_wd_add",
                )
            )
            grad = wd_grad

        update = f"{param}_update"
        if momentum > 0.0:
            # v = momentum * v + grad
            # param = param - lr * v
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
                Node(
                    "Add",
                    [f"{param}_mom_term", grad],
                    [v_new],
                    {},
                    name=f"{param}_mom_add",
                )
            )
            graph.add_node(
                Node(
                    "Mul",
                    [v_new, learning_rate],
                    [update],
                    {},
                    name=f"{param}_lr_mul_mom",
                )
            )
        else:
            graph.add_node(
                Node("Mul", [grad, learning_rate], [update], {}, name=f"{param}_lr_mul")
            )

        graph.add_node(
            Node(
                "Sub", [param, update], [f"{param}_new"], {}, name=f"{param}_update_sub"
            )
        )


def add_adam_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0,
) -> None:
    """Adds Adam optimizer steps to the graph."""
    for param in parameters:
        grad = f"grad_{param}"
        m = f"adam_m_{param}"
        v = f"adam_v_{param}"
        t = f"adam_t_{param}"

        # simplified mock structure
        graph.add_node(
            Node(
                "AdamStep",
                [param, grad, m, v, t, learning_rate],
                [f"{param}_new", f"{m}_new", f"{v}_new", f"{t}_new"],
                {
                    "beta1": beta1,
                    "beta2": beta2,
                    "epsilon": epsilon,
                    "weight_decay": weight_decay,
                },
                name=f"{param}_adam",
            )
        )


def add_adamw_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.01,
) -> None:
    """Adds AdamW optimizer steps to the graph."""
    # Same as Adam but weight decay is applied directly to param
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
                {
                    "beta1": beta1,
                    "beta2": beta2,
                    "epsilon": epsilon,
                    "weight_decay": weight_decay,
                },
                name=f"{param}_adamw",
            )
        )


def add_rmsprop_optimizer(
    graph: Graph,
    learning_rate: str,
    parameters: list[str],
    alpha: float = 0.99,
    epsilon: float = 1e-8,
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
    epsilon: float = 1e-6,
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
    """Adds gradient clipping logic."""
    return
