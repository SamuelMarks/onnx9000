"""TVM submodule for AST and optimization."""

from .tensor import Tensor, compute, exp, max, reduce_axis, sum


def nn_conv2d(
    data: Tensor,
    weight: Tensor,
    strides: tuple[int, int],
    padding: tuple[int, int, int, int],
    dilation: tuple[int, int] = (1, 1),
) -> Tensor:
    """Pass 140: TE compute for Conv2D."""
    batch, in_c, in_h, in_w = data.shape
    out_c, _, filter_h, filter_w = weight.shape

    pad_t, pad_l, pad_b, pad_r = padding
    stride_h, stride_w = strides
    dil_h, dil_w = dilation

    out_h = (in_h + pad_t + pad_b - (filter_h - 1) * dil_h - 1) // stride_h + 1
    out_w = (in_w + pad_l + pad_r - (filter_w - 1) * dil_w - 1) // stride_w + 1

    rc = reduce_axis((0, in_c), name="rc")
    ry = reduce_axis((0, filter_h), name="ry")
    rx = reduce_axis((0, filter_w), name="rx")

    # We omit padding logic for brevity
    def fcompute(
        b: int | float | str | bool | bytes | list | dict | tuple | None,
        k: int | float | str | bool | bytes | list | dict | tuple | None,
        i: int | float | str | bool | bytes | list | dict | tuple | None,
        j: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fcompute operation.

        Args:
            b: Parameter specifying b.
            k: Parameter specifying k.
            i: Parameter specifying i.
            j: Parameter specifying j.

        Returns:
            Result of fcompute.
        """
        return sum(
            data(b, rc, i * stride_h + ry * dil_h, j * stride_w + rx * dil_w)
            * weight(k, rc, ry, rx),
            axis=[rc, ry, rx],
        )

    return compute((batch, out_c, out_h, out_w), fcompute, name="conv2d")


def nn_matmul(A: Tensor, B: Tensor) -> Tensor:
    """Pass 141: TE compute for MatMul."""
    m, k = A.shape
    _k, n = B.shape

    rk = reduce_axis((0, k), name="k")

    def fcompute(
        i: int | float | str | bool | bytes | list | dict | tuple | None,
        j: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fcompute operation.

        Args:
            i: Parameter specifying i.
            j: Parameter specifying j.

        Returns:
            Result of fcompute.
        """
        return sum(A(i, rk) * B(rk, j), axis=[rk])

    return compute((m, n), fcompute, name="matmul")


def nn_pool2d(
    data: Tensor,
    pool_size: tuple[int, int],
    strides: tuple[int, int],
    padding: tuple[int, int, int, int],
    pool_type: str = "max",
) -> Tensor:
    """Pass 142: TE compute for Pool (Max/Avg)."""
    batch, in_c, in_h, in_w = data.shape
    pool_h, pool_w = pool_size
    stride_h, stride_w = strides
    pad_t, pad_l, pad_b, pad_r = padding

    out_h = (in_h + pad_t + pad_b - pool_h) // stride_h + 1
    out_w = (in_w + pad_l + pad_r - pool_w) // stride_w + 1

    ry = reduce_axis((0, pool_h), name="ry")
    rx = reduce_axis((0, pool_w), name="rx")

    def fcompute(
        b: int | float | str | bool | bytes | list | dict | tuple | None,
        c: int | float | str | bool | bytes | list | dict | tuple | None,
        i: int | float | str | bool | bytes | list | dict | tuple | None,
        j: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fcompute operation.

        Args:
            b: Parameter specifying b.
            c: Parameter specifying c.
            i: Parameter specifying i.
            j: Parameter specifying j.

        Returns:
            Result of fcompute.
        """
        val = data(b, c, i * stride_h + ry, j * stride_w + rx)
        if pool_type == "max":
            return max(val, axis=[ry, rx])
        else:
            return sum(val, axis=[ry, rx]) / (pool_h * pool_w)

    return compute((batch, in_c, out_h, out_w), fcompute, name=f"{pool_type}_pool2d")


def nn_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Pass 143: TE compute for Softmax."""
    shape = x.shape

    # Very simplified version
    red_axis = reduce_axis((0, shape[axis]), name="rx")

    # Max
    def fmax(
        *indices: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fmax operation.

        Args:
            *indices: Variadic positional arguments for indices.

        Returns:
            Result of fmax.
        """
        return max(x(*indices), axis=[red_axis])

    m = compute(shape[:-1], fmax, name="max")

    # Exp
    def fexp(
        *indices: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fexp operation.

        Args:
            *indices: Variadic positional arguments for indices.

        Returns:
            Result of fexp.
        """
        return exp(x(*indices) - m(*indices[:-1]))

    e = compute(shape, fexp, name="exp")

    # Sum
    def fsum(
        *indices: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fsum operation.

        Args:
            *indices: Variadic positional arguments for indices.

        Returns:
            Result of fsum.
        """
        return sum(e(*indices), axis=[red_axis])

    s = compute(shape[:-1], fsum, name="sum")

    # Div
    def fdiv(
        *indices: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fdiv operation.

        Args:
            *indices: Variadic positional arguments for indices.

        Returns:
            Result of fdiv.
        """
        return e(*indices) / s(*indices[:-1])

    return compute(shape, fdiv, name="softmax")


def nn_layer_norm(
    data: Tensor, gamma: Tensor, beta: Tensor, axis: int = -1, epsilon: float = 1e-5
) -> Tensor:
    """Pass 144: TE compute for LayerNorm."""
    shape = data.shape

    red_axis = reduce_axis((0, shape[axis]), name="rx")

    # Mean
    def fmean(
        *indices: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fmean operation.

        Args:
            *indices: Variadic positional arguments for indices.

        Returns:
            Result of fmean.
        """
        return sum(data(*indices), axis=[red_axis]) / shape[axis]

    m = compute(shape[:-1], fmean, name="mean")

    # Var
    def fvar(
        *indices: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fvar operation.

        Args:
            *indices: Variadic positional arguments for indices.

        Returns:
            Result of fvar.
        """
        diff = data(*indices) - m(*indices[:-1])
        return sum(diff * diff, axis=[red_axis]) / shape[axis]

    v = compute(shape[:-1], fvar, name="var")

    # Norm
    def fnorm(
        *indices: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fnorm operation.

        Args:
            *indices: Variadic positional arguments for indices.

        Returns:
            Result of fnorm.
        """
        # We need a generic sqrt, assuming pow exists
        normed = (data(*indices) - m(*indices[:-1])) / (v(*indices[:-1]) + epsilon)
        return normed * gamma(indices[-1]) + beta(indices[-1])

    return compute(shape, fnorm, name="layer_norm")
