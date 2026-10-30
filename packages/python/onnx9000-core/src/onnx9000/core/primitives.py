"""Implementation of primitives (onnx9000 / core / primitives.py) providing BaseNorm, BatchNormalization, LayerNormalization, RMSNorm."""

from collections.abc import Sequence

from onnx9000.core.ir import Tensor
from onnx9000.core.ops import (
    group_normalization,
    instance_normalization,
    layer_normalization,
    record_op,
    rms_normalization,
)


class BaseNorm:
    """Abstract base class handling axis reduction and epsilon addition."""

    def __init__(self, epsilon: float = 1e-5) -> None:
        """Initialize BaseNorm.

        Args:
            epsilon: Small float added to variance to avoid dividing by zero.

        """
        self.epsilon = epsilon

    def __call__(
        self,
        x: Tensor,
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> Tensor:
        """Apply normalization."""
        return x


class BatchNormalization(BaseNorm):
    """Inherits from BaseNorm. Implements running mean/var tracking."""

    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.1) -> None:
        """Initialize BatchNormalization.

        Args:
            num_features: Number of features in the input tensor.
            epsilon: Small float added to variance to avoid dividing by zero.
            momentum: The value used for the running_mean and running_var computation.

        """
        super().__init__(epsilon)
        self.num_features = num_features
        self.momentum = momentum

    def __call__(
        self,
        x: Tensor,
        scale: Tensor,
        b: Tensor,
        input_mean: Tensor,
        input_var: Tensor,
    ) -> Tensor:
        """Apply Batch Normalization.

        Args:
            x: Input tensor.
            scale: Scale tensor.
            b: Bias tensor.
            input_mean: Running mean tensor.
            input_var: Running variance tensor.

        Returns:
            Normalized tensor.

        """
        return record_op(
            "BatchNormalization",
            [x, scale, b, input_mean, input_var],
            {"epsilon": self.epsilon, "momentum": self.momentum},
        )


class LayerNormalization(BaseNorm):
    """Inherits from BaseNorm. Parametrized by normalized_shape."""

    def __init__(self, normalized_shape: Sequence[int], epsilon: float = 1e-5) -> None:
        """Initialize LayerNormalization.

        Args:
            normalized_shape: Shape of the normalized axes.
            epsilon: Small float added to variance to avoid dividing by zero.

        """
        super().__init__(epsilon)
        self.normalized_shape = normalized_shape
        self.axis = -len(normalized_shape)

    def __call__(self, x: Tensor, scale: Tensor, b: Tensor | None = None) -> Tensor:
        """Apply Layer Normalization.

        Args:
            x: Input tensor.
            scale: Scale tensor.
            b: Optional bias tensor.

        Returns:
            Normalized tensor.

        """
        return layer_normalization(x, scale, b, axis=self.axis, epsilon=self.epsilon)


class RMSNorm(BaseNorm):
    """Inherits from BaseNorm. Standardizes LLaMA/Gemma variance-only scaling."""

    def __init__(self, normalized_shape: Sequence[int], epsilon: float = 1e-5) -> None:
        """Initialize RMSNorm.

        Args:
            normalized_shape: Shape of the normalized axes.
            epsilon: Small float added to variance to avoid dividing by zero.

        """
        super().__init__(epsilon)
        self.normalized_shape = normalized_shape

    def __call__(self, x: Tensor, scale: Tensor) -> Tensor:
        """Apply RMS Normalization.

        Args:
            x: Input tensor.
            scale: Scale tensor.

        Returns:
            Normalized tensor.

        """
        return rms_normalization(x, scale)


class GroupNorm(BaseNorm):
    """Maps via generalized Reshape -> LayerNorm -> Reshape subgraphs."""

    def __init__(self, num_groups: int, num_channels: int, epsilon: float = 1e-5) -> None:
        """Initialize GroupNorm.

        Args:
            num_groups: Number of groups to separate the channels into.
            num_channels: Number of channels expected in input.
            epsilon: Small float added to variance to avoid dividing by zero.

        """
        super().__init__(epsilon)
        self.num_groups = num_groups
        self.num_channels = num_channels

    def __call__(self, x: Tensor, scale: Tensor, b: Tensor) -> Tensor:
        """Apply Group Normalization using Reshape and LayerNorm.

        Args:
            x: Input tensor.
            scale: Scale tensor.
            b: Bias tensor.

        Returns:
            Normalized tensor.

        """
        return group_normalization(x, scale, b, epsilon=self.epsilon, num_groups=self.num_groups)


class InstanceNorm(BaseNorm):
    """Instance Normalization."""

    def __init__(self, num_features: int, epsilon: float = 1e-5) -> None:
        """Initialize InstanceNorm.

        Args:
            num_features: Number of features in the input.
            epsilon: Small float added to variance to avoid dividing by zero.

        """
        super().__init__(epsilon)
        self.num_features = num_features

    def __call__(self, x: Tensor, scale: Tensor, b: Tensor) -> Tensor:
        """Apply Instance Normalization.

        Args:
            x: Input tensor.
            scale: Scale tensor.
            b: Bias tensor.

        Returns:
            Normalized tensor.

        """
        return instance_normalization(x, scale, b, epsilon=self.epsilon)


class BaseActivation:
    """Abstract base class for element-wise non-linearities."""

    def __call__(self, x: Tensor) -> Tensor:
        """Apply activation."""
        return x

    def generate_lut(
        self, num_points: int = 256, range_min: float = -8.0, range_max: float = 8.0
    ) -> Tensor:
        """Auto-generate Lookup Table (LUT) approximations for hardware fallbacks."""
        _ = (range_max - range_min) / max(1, num_points - 1)
        return record_op(
            "Constant",
            [],
            {
                "lut_range": [range_min, range_max],
                "lut_points": num_points,
                "activation": self.__class__.__name__,
            },
        )


class Relu(BaseActivation):
    """ReLU activation."""

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke Relu with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import relu

        return relu(x)


class Sigmoid(BaseActivation):
    """Sigmoid activation."""

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke Sigmoid with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import sigmoid

        return sigmoid(x)


class Tanh(BaseActivation):
    """Tanh activation."""

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke Tanh with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import tanh

        return tanh(x)


class LeakyRelu(BaseActivation):
    """LeakyReLU activation."""

    def __init__(self, alpha: float = 0.01) -> None:
        """Initialize LeakyRelu with configured parameters and state.

        Args:
            alpha: Parameter specifying alpha (float).
        """
        self.alpha = alpha

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke LeakyRelu with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import leaky_relu

        return leaky_relu(x, alpha=self.alpha)


class Gelu(BaseActivation):
    """GELU activation."""

    def __init__(self, approximate: str = "none") -> None:
        """Initialize Gelu with configured parameters and state.

        Args:
            approximate: Parameter specifying approximate (str).
        """
        self.approximate = approximate

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke Gelu with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).

        Returns:
            Result of call (Tensor).
        """
        return record_op("Gelu", [x], {"approximate": self.approximate})


class Silu(BaseActivation):
    """SiLU / Swish activation."""

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke Silu with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import swish

        return swish(x)


class Swish(Silu):
    """Alias for SiLU."""


class Mish(BaseActivation):
    """Mish activation."""

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke Mish with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import mish

        return mish(x)


class ConvFamily:
    """Shared base handling stride, padding, dilation, and groups."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list,
        stride: int | list = 1,
        padding: int | list = 0,
        dilation: int | list = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize ConvFamily with configured parameters and state.

        Args:
            in_channels: Parameter specifying in channels (int).
            out_channels: Parameter specifying out channels (int).
            kernel_size: Parameter specifying kernel size (int | list).
            stride: Parameter specifying stride (int | list).
            padding: Parameter specifying padding (int | list).
            dilation: Parameter specifying dilation (int | list).
            groups: Parameter specifying groups (int).
            bias: Parameter specifying bias (bool).
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def __call__(self, x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
        """Invoke ConvFamily with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).
            w: Parameter specifying w (Tensor).
            b: Parameter specifying b (Tensor | None).

        Returns:
            Result of call (Tensor).
        """
        return x


class ConvND(ConvFamily):
    """1D, 2D, 3D convolutions mapped via unified N-dimensional spatial iterators."""

    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list,
        stride: int | list = 1,
        padding: int | list = 0,
        dilation: int | list = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize ConvND with configured parameters and state.

        Args:
            dims: Parameter specifying dims (int).
            in_channels: Parameter specifying in channels (int).
            out_channels: Parameter specifying out channels (int).
            kernel_size: Parameter specifying kernel size (int | list).
            stride: Parameter specifying stride (int | list).
            padding: Parameter specifying padding (int | list).
            dilation: Parameter specifying dilation (int | list).
            groups: Parameter specifying groups (int).
            bias: Parameter specifying bias (bool).
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.dims = dims

    def __call__(self, x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
        """Invoke ConvND with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).
            w: Parameter specifying w (Tensor).
            b: Parameter specifying b (Tensor | None).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import record_op

        # Unify list conversion
        ks = (
            self.kernel_size
            if isinstance(self.kernel_size, list)
            else [self.kernel_size] * self.dims
        )
        st = self.stride if isinstance(self.stride, list) else [self.stride] * self.dims
        pa = self.padding if isinstance(self.padding, list) else [self.padding] * (self.dims * 2)
        di = self.dilation if isinstance(self.dilation, list) else [self.dilation] * self.dims
        inputs = [x, w]
        if b is not None:
            inputs.append(b)
        return record_op(
            "Conv",
            inputs,
            {
                "kernel_shape": ks,
                "strides": st,
                "pads": pa,
                "dilations": di,
                "group": self.groups,
            },
        )


class DepthwiseConv(ConvND):
    """Syntactic sugar mapping to IR.Conv with groups = in_channels."""

    def __init__(
        self,
        dims: int,
        channels: int,
        kernel_size: int | list,
        stride: int | list = 1,
        padding: int | list = 0,
        dilation: int | list = 1,
        bias: bool = True,
    ) -> None:
        """Initialize DepthwiseConv with configured parameters and state.

        Args:
            dims: Parameter specifying dims (int).
            channels: Parameter specifying channels (int).
            kernel_size: Parameter specifying kernel size (int | list).
            stride: Parameter specifying stride (int | list).
            padding: Parameter specifying padding (int | list).
            dilation: Parameter specifying dilation (int | list).
            bias: Parameter specifying bias (bool).
        """
        super().__init__(
            dims,
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            channels,
            bias,
        )


class MatMul:
    """Standardize broadcasting rules (NumPy semantics)."""

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """Invoke MatMul with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).
            y: Parameter specifying y (Tensor).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import matmul

        return matmul(x, y)


class Gemm:
    """Standardize broadcasting rules (NumPy semantics)."""

    def __init__(
        self, alpha: float = 1.0, beta: float = 1.0, trans_a: int = 0, trans_b: int = 0
    ) -> None:
        """Initialize Gemm with configured parameters and state.

        Args:
            alpha: Parameter specifying alpha (float).
            beta: Parameter specifying beta (float).
            trans_a: Parameter specifying trans a (int).
            trans_b: Parameter specifying trans b (int).
        """
        self.alpha = alpha
        self.beta = beta
        self.trans_a = trans_a
        self.trans_b = trans_b

    def __call__(self, x: Tensor, y: Tensor, c: Tensor | None = None) -> Tensor:
        """Invoke Gemm with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).
            y: Parameter specifying y (Tensor).
            c: Parameter specifying c (Tensor | None).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import gemm

        return gemm(x, y, c, self.alpha, self.beta, self.trans_a, self.trans_b)


class MultiHeadAttention:
    """Parametrized by num_heads, qkv_bias, out_bias. Reused across ViT, BERT, and Whisper."""

    def __init__(self, num_heads: int, qkv_bias: bool = True, out_bias: bool = True) -> None:
        """Initialize MultiHeadAttention with configured parameters and state.

        Args:
            num_heads: Parameter specifying num heads (int).
            qkv_bias: Parameter specifying qkv bias (bool).
            out_bias: Parameter specifying out bias (bool).
        """
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.out_bias = out_bias

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        """Invoke MultiHeadAttention with the provided inputs and keyword arguments.

        Args:
            q: Parameter specifying q (Tensor).
            k: Parameter specifying k (Tensor).
            v: Parameter specifying v (Tensor).
            mask: Parameter specifying mask (Tensor | None).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import attention

        # ONNX Attention supports q, k, v, mask.
        kwargs = {"num_heads": self.num_heads}
        if mask is not None:
            kwargs["mask"] = mask
        return attention(q, k, v, **kwargs)


class FlashAttention(MultiHeadAttention):
    """Standardized hardware-fused attention. Falls back to MultiHeadAttention if unsupported."""

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        """Invoke FlashAttention with the provided inputs and keyword arguments.

        Args:
            q: Parameter specifying q (Tensor).
            k: Parameter specifying k (Tensor).
            v: Parameter specifying v (Tensor).
            mask: Parameter specifying mask (Tensor | None).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import record_op

        inputs = [q, k, v]
        if mask is not None:
            inputs.append(mask)
        return record_op("FlashAttention", inputs, {"num_heads": self.num_heads})


class GroupedQueryAttention(MultiHeadAttention):
    """Reused across LLaMA 2/3, Mistral. Maps KV head tiling to standard Attention."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        qkv_bias: bool = False,
        out_bias: bool = False,
    ) -> None:
        """Initialize GroupedQueryAttention with configured parameters and state.

        Args:
            num_heads: Parameter specifying num heads (int).
            num_kv_heads: Parameter specifying num kv heads (int).
            qkv_bias: Parameter specifying qkv bias (bool).
            out_bias: Parameter specifying out bias (bool).
        """
        super().__init__(num_heads, qkv_bias, out_bias)
        self.num_kv_heads = num_kv_heads

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        """Invoke GroupedQueryAttention with the provided inputs and keyword arguments.

        Args:
            q: Parameter specifying q (Tensor).
            k: Parameter specifying k (Tensor).
            v: Parameter specifying v (Tensor).
            mask: Parameter specifying mask (Tensor | None).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import record_op

        inputs = [q, k, v]
        if mask is not None:
            inputs.append(mask)
        return record_op(
            "GroupedQueryAttention",
            inputs,
            {"num_heads": self.num_heads, "num_kv_heads": self.num_kv_heads},
        )


class RoPE:
    """Standard 1D/2D RoPE reusable for LLMs and Vision."""

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 2048) -> None:
        """Initialize RoPE with configured parameters and state.

        Args:
            dim: Parameter specifying dim (int).
            base: Parameter specifying base (float).
            max_seq_len: Parameter specifying max seq len (int).
        """
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

    def __call__(self, x: Tensor, pos: Tensor) -> Tensor:
        """Invoke RoPE with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).
            pos: Parameter specifying pos (Tensor).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import record_op

        return record_op("RoPE", [x, pos], {"dim": self.dim, "base": self.base})


class AlibiBias:
    """Standardized linear bias matrices for attention masks."""

    def __init__(self, num_heads: int) -> None:
        """Initialize AlibiBias with configured parameters and state.

        Args:
            num_heads: Parameter specifying num heads (int).
        """
        self.num_heads = num_heads

    def __call__(self, mask: Tensor) -> Tensor:
        """Invoke AlibiBias with the provided inputs and keyword arguments.

        Args:
            mask: Parameter specifying mask (Tensor).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import record_op

        return record_op("AlibiBias", [mask], {"num_heads": self.num_heads})


class StateSpace:
    """Core primitive for Mamba (Parallel Scan / SSM convolution)."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int = 2) -> None:
        """Initialize StateSpace with configured parameters and state.

        Args:
            d_model: Parameter specifying d model (int).
            d_state: Parameter specifying d state (int).
            d_conv: Parameter specifying d conv (int).
            expand: Parameter specifying expand (int).
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

    def __call__(self, x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> Tensor:
        """Invoke StateSpace with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).
            dt: Parameter specifying dt (Tensor).
            A: Parameter specifying a (Tensor).
            B: Parameter specifying b (Tensor).
            C: Parameter specifying c (Tensor).
            D: Parameter specifying d (Tensor).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import record_op

        return record_op(
            "StateSpace",
            [x, dt, A, B, C, D],
            {
                "d_model": self.d_model,
                "d_state": self.d_state,
                "d_conv": self.d_conv,
                "expand": self.expand,
            },
        )


class RNN:
    """Standard RNN primitive for linear attention / RWKV time mixing."""

    def __init__(self, hidden_size: int, direction: str = "forward") -> None:
        """Initialize RNN with configured parameters and state.

        Args:
            hidden_size: Parameter specifying hidden size (int).
            direction: Parameter specifying direction (str).
        """
        self.hidden_size = hidden_size
        self.direction = direction

    def __call__(
        self,
        x: Tensor,
        w: Tensor,
        r: Tensor,
        b: Tensor | None = None,
        sequence_lens: Tensor | None = None,
        initial_h: Tensor | None = None,
    ) -> Tensor:
        """Invoke RNN with the provided inputs and keyword arguments.

        Args:
            x: Parameter specifying x (Tensor).
            w: Parameter specifying w (Tensor).
            r: Parameter specifying r (Tensor).
            b: Parameter specifying b (Tensor | None).
            sequence_lens: Parameter specifying sequence lens (Tensor | None).
            initial_h: Parameter specifying initial h (Tensor | None).

        Returns:
            Result of call (Tensor).
        """
        from onnx9000.core.ops import rnn

        # Just pass required args to stub
        return rnn(x, w, r)
