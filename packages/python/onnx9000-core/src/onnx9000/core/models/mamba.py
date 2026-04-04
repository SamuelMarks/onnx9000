"""Module docstring."""

from typing import Any

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.ops import add, mul
from onnx9000.core.primitives import ConvND, Gemm, RMSNorm, Silu, StateSpace


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:  # noqa: D103
    return Variable(name=name, shape=shape, dtype=dtype)


class MambaBlock:  # noqa: D101
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, prefix: str = ""):  # noqa: D107
        self.prefix = prefix
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.norm = RMSNorm((d_model,))
        self.in_proj = Gemm(trans_b=1)

        self.conv1d = ConvND(
            1,
            d_model * expand,
            d_model * expand,
            kernel_size=d_conv,
            groups=d_model * expand,
            padding=d_conv - 1,
        )
        self.act = Silu()

        self.x_proj = Gemm(trans_b=1)
        self.dt_proj = Gemm(trans_b=1)

        self.ssm = StateSpace(d_model, d_state, d_conv, expand)

        self.out_proj = Gemm(trans_b=1)

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        identity = x
        x_norm = self.norm(x, get_param(f"{self.prefix}.norm.weight", [self.d_model]))

        # in_proj outputs [B, L, 2 * d_inner]
        xz = self.in_proj(
            x_norm,
            get_param(
                f"{self.prefix}.in_proj.weight", [self.d_model * self.expand * 2, self.d_model]
            ),
        )

        # Split into x and z
        from onnx9000.core.ops import constant
        from onnx9000.core.ops import slice as slice_op

        d_inner = self.d_model * self.expand
        x_inner = slice_op(
            xz, constant([0], dtype=7), constant([d_inner], dtype=7), constant([2], dtype=7)
        )
        z = slice_op(
            xz,
            constant([d_inner], dtype=7),
            constant([d_inner * 2], dtype=7),
            constant([2], dtype=7),
        )

        # Conv1D needs [B, C, L]
        from onnx9000.core.ops import transpose

        x_inner_t = transpose(x_inner, perm=[0, 2, 1])
        x_inner_conv = self.conv1d(
            x_inner_t,
            get_param(f"{self.prefix}.conv1d.weight", [d_inner, 1, self.d_conv]),
            get_param(f"{self.prefix}.conv1d.bias", [d_inner]),
        )
        x_inner_conv = transpose(x_inner_conv, perm=[0, 2, 1])

        # Slice to original L since causal padding adds d_conv - 1
        from onnx9000.core.ops import gather, shape

        seq_len = gather(shape(x), constant([1], dtype=7), axis=0)
        x_inner_conv = slice_op(
            x_inner_conv, constant([0], dtype=7), seq_len, constant([1], dtype=7)
        )

        x_act = self.act(x_inner_conv)

        # SSM parameters
        x_dt_B_C = self.x_proj(
            x_act, get_param(f"{self.prefix}.x_proj.weight", [self.d_state * 2 + 1, d_inner])
        )
        dt = slice_op(
            x_dt_B_C, constant([0], dtype=7), constant([1], dtype=7), constant([2], dtype=7)
        )
        B = slice_op(
            x_dt_B_C,
            constant([1], dtype=7),
            constant([self.d_state + 1], dtype=7),
            constant([2], dtype=7),
        )
        C = slice_op(
            x_dt_B_C,
            constant([self.d_state + 1], dtype=7),
            constant([self.d_state * 2 + 1], dtype=7),
            constant([2], dtype=7),
        )

        dt = self.dt_proj(
            dt,
            get_param(f"{self.prefix}.dt_proj.weight", [d_inner, 1]),
            get_param(f"{self.prefix}.dt_proj.bias", [d_inner]),
        )

        A = get_param(f"{self.prefix}.A_log", [d_inner, self.d_state])
        D = get_param(f"{self.prefix}.D", [d_inner])

        # State Space Op
        y = self.ssm(x_act, dt, A, B, C, D)

        # Gate
        z_act = self.act(z)
        y = mul(y, z_act)

        # Out
        out = self.out_proj(y, get_param(f"{self.prefix}.out_proj.weight", [self.d_model, d_inner]))
        return add(identity, out)


class Mamba:  # noqa: D101
    def __init__(  # noqa: D107
        self,
        vocab_size: int = 50277,
        d_model: int = 768,
        n_layer: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.blocks = [
            MambaBlock(d_model, d_state, d_conv, expand, prefix=f"blocks.{i}")
            for i in range(n_layer)
        ]
        self.norm = RMSNorm((d_model,))
        self.lm_head = Gemm(trans_b=1)

    def __call__(self, input_ids: Tensor) -> Tensor:  # noqa: D102
        from onnx9000.core.ops import gather

        x = gather(
            get_param("embedding.weight", [self.vocab_size, self.d_model]), input_ids, axis=0
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x, get_param("norm.weight", [self.d_model]))
        x = self.lm_head(x, get_param("lm_head.weight", [self.vocab_size, self.d_model]))
        return x


def mamba_130m(**kwargs: Any) -> Mamba:  # noqa: D103
    return Mamba(
        vocab_size=50277, d_model=768, n_layer=24, d_state=16, d_conv=4, expand=2, **kwargs
    )
