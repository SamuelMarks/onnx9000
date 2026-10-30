"""Tests for PyTorch-compatible frontend operations in onnx9000.converters.torch_like."""

from __future__ import annotations

import numpy as np
from onnx9000.converters import torch_like
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Tensor as IRTensor


def test_tensor_creation_and_dtypes() -> None:
    """Verify tensor, zeros, ones, randn, and dtype constants."""
    t_default = torch_like.tensor([1.0, 2.0, 3.0])
    assert t_default.shape == (3,)
    assert t_default.dtype == DType.FLOAT64
    assert t_default.tolist() == [1.0, 2.0, 3.0]

    t_int = torch_like.tensor([1, 2], dtype=torch_like.int64)
    assert t_int.dtype == DType.INT64
    assert t_int.tolist() == [1, 2]

    z = torch_like.zeros(2, 3)
    assert z.shape == (2, 3)
    assert z.dtype == DType.FLOAT32
    assert np.all(z.numpy() == 0.0)

    o = torch_like.ones(2, 2, dtype=torch_like.float64)
    assert o.shape == (2, 2)
    assert o.dtype == DType.FLOAT64
    assert np.all(o.numpy() == 1.0)

    r = torch_like.randn(3, 4)
    assert r.shape == (3, 4)
    assert r.dtype == DType.FLOAT32

    assert torch_like.float32 == DType.FLOAT32
    assert torch_like.float64 == DType.FLOAT64
    assert torch_like.int32 == DType.INT32
    assert torch_like.int64 == DType.INT64
    assert torch_like.bool == DType.BOOL


def test_jit_and_onnx_export(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify jit.trace, jit.script, and onnx.export produce valid ONNX graphs."""

    def add_fn(x: torch_like.Tensor) -> torch_like.Tensor:
        """Execute add fn operation.

        Args:
            x: Parameter specifying x (torch_like.Tensor).

        Returns:
            Result of add fn (torch_like.Tensor).
        """
        return x + x

    example = torch_like.zeros(2, 2)
    traced = torch_like.jit.trace(add_fn, example)
    graph = traced.to_graph()
    assert len(graph.nodes) == 1
    assert graph.nodes[0].op_type == "Add"

    scripted = torch_like.jit.script(add_fn)
    s_graph = scripted.to_graph()
    assert s_graph.name == "add_fn"

    out_file = tmp_path / "model.onnx"
    torch_like.onnx.export(
        add_fn,
        example,
        str(out_file),
        input_names=["input_x"],
        output_names=["output_y"],
    )
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_matmul() -> None:
    """Verify matmul delegates to core MatMul operator."""
    a = IRTensor(name="a", shape=(2, 3), dtype=DType.FLOAT32)
    b = IRTensor(name="b", shape=(3, 4), dtype=DType.FLOAT32)
    out = torch_like.matmul(a, b)
    assert isinstance(out, IRTensor)
    assert out.name == "MatMul_out"
    assert out.dtype == DType.FLOAT32


def test_rand() -> None:
    """Verify torch_like.rand records a symbolic Rand_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rand(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rand_out"
    assert out.dtype == DType.FLOAT32


def test_arange() -> None:
    """Verify torch_like.arange records a symbolic Arange_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arange(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arange_out"
    assert out.dtype == DType.FLOAT32


def test_as_tensor() -> None:
    """Verify torch_like.as_tensor records a symbolic As_tensor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.as_tensor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "As_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_asarray() -> None:
    """Verify torch_like.asarray records a symbolic Asarray_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.asarray(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Asarray_out"
    assert out.dtype == DType.FLOAT32


def test_bartlett_window() -> None:
    """Verify torch_like.bartlett_window records a symbolic Bartlett_window_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bartlett_window(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bartlett_window_out"
    assert out.dtype == DType.FLOAT32


def test_binomial() -> None:
    """Verify torch_like.binomial records a symbolic Binomial_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.binomial(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Binomial_out"
    assert out.dtype == DType.FLOAT32


def test_empty() -> None:
    """Verify torch_like.empty records a symbolic Empty_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.empty(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Empty_out"
    assert out.dtype == DType.FLOAT32


def test_empty_like() -> None:
    """Verify torch_like.empty_like records a symbolic Empty_like_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.empty_like(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Empty_like_out"
    assert out.dtype == DType.FLOAT32


def test_empty_permuted() -> None:
    """Verify torch_like.empty_permuted records a symbolic Empty_permuted_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.empty_permuted(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Empty_permuted_out"
    assert out.dtype == DType.FLOAT32


def test_empty_quantized() -> None:
    """Verify torch_like.empty_quantized records a symbolic Empty_quantized_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.empty_quantized(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Empty_quantized_out"
    assert out.dtype == DType.FLOAT32


def test_empty_strided() -> None:
    """Verify torch_like.empty_strided records a symbolic Empty_strided_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.empty_strided(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Empty_strided_out"
    assert out.dtype == DType.FLOAT32


def test_eye() -> None:
    """Verify torch_like.eye records a symbolic Eye_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.eye(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Eye_out"
    assert out.dtype == DType.FLOAT32


def test_fill() -> None:
    """Verify torch_like.fill records a symbolic Fill_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fill(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fill_out"
    assert out.dtype == DType.FLOAT32


def test_fill_() -> None:
    """Verify torch_like.fill_ records a symbolic Fill__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fill_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fill__out"
    assert out.dtype == DType.FLOAT32


def test_from_file() -> None:
    """Verify torch_like.from_file records a symbolic From_file_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.from_file(x)
    assert isinstance(out, IRTensor)
    assert out.name == "From_file_out"
    assert out.dtype == DType.FLOAT32


def test_from_numpy() -> None:
    """Verify torch_like.from_numpy records a symbolic From_numpy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.from_numpy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "From_numpy_out"
    assert out.dtype == DType.FLOAT32


def test_frombuffer() -> None:
    """Verify torch_like.frombuffer records a symbolic Frombuffer_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.frombuffer(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Frombuffer_out"
    assert out.dtype == DType.FLOAT32


def test_full() -> None:
    """Verify torch_like.full records a symbolic Full_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.full(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Full_out"
    assert out.dtype == DType.FLOAT32


def test_full_like() -> None:
    """Verify torch_like.full_like records a symbolic Full_like_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.full_like(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Full_like_out"
    assert out.dtype == DType.FLOAT32


def test_hamming_window() -> None:
    """Verify torch_like.hamming_window records a symbolic Hamming_window_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.hamming_window(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Hamming_window_out"
    assert out.dtype == DType.FLOAT32


def test_hann_window() -> None:
    """Verify torch_like.hann_window records a symbolic Hann_window_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.hann_window(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Hann_window_out"
    assert out.dtype == DType.FLOAT32


def test_kaiser_window() -> None:
    """Verify torch_like.kaiser_window records a symbolic Kaiser_window_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.kaiser_window(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Kaiser_window_out"
    assert out.dtype == DType.FLOAT32


def test_linspace() -> None:
    """Verify torch_like.linspace records a symbolic Linspace_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.linspace(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Linspace_out"
    assert out.dtype == DType.FLOAT32


def test_logspace() -> None:
    """Verify torch_like.logspace records a symbolic Logspace_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.logspace(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Logspace_out"
    assert out.dtype == DType.FLOAT32


def test_normal() -> None:
    """Verify torch_like.normal records a symbolic Normal_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.normal(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Normal_out"
    assert out.dtype == DType.FLOAT32


def test_ones_like() -> None:
    """Verify torch_like.ones_like records a symbolic Ones_like_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ones_like(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ones_like_out"
    assert out.dtype == DType.FLOAT32


def test_poisson() -> None:
    """Verify torch_like.poisson records a symbolic Poisson_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.poisson(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Poisson_out"
    assert out.dtype == DType.FLOAT32


def test_rand_like() -> None:
    """Verify torch_like.rand_like records a symbolic Rand_like_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rand_like(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rand_like_out"
    assert out.dtype == DType.FLOAT32


def test_randint() -> None:
    """Verify torch_like.randint records a symbolic Randint_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.randint(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Randint_out"
    assert out.dtype == DType.FLOAT32


def test_randint_like() -> None:
    """Verify torch_like.randint_like records a symbolic Randint_like_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.randint_like(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Randint_like_out"
    assert out.dtype == DType.FLOAT32


def test_randn_like() -> None:
    """Verify torch_like.randn_like records a symbolic Randn_like_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.randn_like(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Randn_like_out"
    assert out.dtype == DType.FLOAT32


def test_randperm() -> None:
    """Verify torch_like.randperm records a symbolic Randperm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.randperm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Randperm_out"
    assert out.dtype == DType.FLOAT32


def test_range() -> None:
    """Verify torch_like.range records a symbolic Range_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.range(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Range_out"
    assert out.dtype == DType.FLOAT32


def test_scalar_tensor() -> None:
    """Verify torch_like.scalar_tensor records a symbolic Scalar_tensor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.scalar_tensor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Scalar_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_bsc_tensor() -> None:
    """Verify torch_like.sparse_bsc_tensor records a symbolic Sparse_bsc_tensor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sparse_bsc_tensor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sparse_bsc_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_bsr_tensor() -> None:
    """Verify torch_like.sparse_bsr_tensor records a symbolic Sparse_bsr_tensor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sparse_bsr_tensor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sparse_bsr_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_compressed_tensor() -> None:
    """Verify torch_like.sparse_compressed_tensor records a symbolic Sparse_compressed_tensor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sparse_compressed_tensor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sparse_compressed_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_coo_tensor() -> None:
    """Verify torch_like.sparse_coo_tensor records a symbolic Sparse_coo_tensor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sparse_coo_tensor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sparse_coo_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_csc_tensor() -> None:
    """Verify torch_like.sparse_csc_tensor records a symbolic Sparse_csc_tensor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sparse_csc_tensor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sparse_csc_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_csr_tensor() -> None:
    """Verify torch_like.sparse_csr_tensor records a symbolic Sparse_csr_tensor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sparse_csr_tensor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sparse_csr_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_tril_indices() -> None:
    """Verify torch_like.tril_indices records a symbolic Tril_indices_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.tril_indices(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Tril_indices_out"
    assert out.dtype == DType.FLOAT32


def test_triu_indices() -> None:
    """Verify torch_like.triu_indices records a symbolic Triu_indices_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.triu_indices(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Triu_indices_out"
    assert out.dtype == DType.FLOAT32


def test_vander() -> None:
    """Verify torch_like.vander records a symbolic Vander_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.vander(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Vander_out"
    assert out.dtype == DType.FLOAT32


def test_zero_() -> None:
    """Verify torch_like.zero_ records a symbolic Zero__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.zero_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Zero__out"
    assert out.dtype == DType.FLOAT32


def test_zeros_like() -> None:
    """Verify torch_like.zeros_like records a symbolic Zeros_like_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.zeros_like(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Zeros_like_out"
    assert out.dtype == DType.FLOAT32


def test_lobpcg() -> None:
    """Verify torch_like.lobpcg records a symbolic Lobpcg_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.lobpcg(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Lobpcg_out"
    assert out.dtype == DType.FLOAT32


def test_addbmm() -> None:
    """Verify torch_like.addbmm records a symbolic Addbmm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.addbmm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Addbmm_out"
    assert out.dtype == DType.FLOAT32


def test_addmm() -> None:
    """Verify torch_like.addmm records a symbolic Addmm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.addmm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Addmm_out"
    assert out.dtype == DType.FLOAT32


def test_addmv() -> None:
    """Verify torch_like.addmv records a symbolic Addmv_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.addmv(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Addmv_out"
    assert out.dtype == DType.FLOAT32


def test_addmv_() -> None:
    """Verify torch_like.addmv_ records a symbolic Addmv__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.addmv_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Addmv__out"
    assert out.dtype == DType.FLOAT32


def test_addr() -> None:
    """Verify torch_like.addr records a symbolic Addr_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.addr(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Addr_out"
    assert out.dtype == DType.FLOAT32


def test_adjoint() -> None:
    """Verify torch_like.adjoint records a symbolic Adjoint_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.adjoint(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Adjoint_out"
    assert out.dtype == DType.FLOAT32


def test_baddbmm() -> None:
    """Verify torch_like.baddbmm records a symbolic Baddbmm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.baddbmm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Baddbmm_out"
    assert out.dtype == DType.FLOAT32


def test_bmm() -> None:
    """Verify torch_like.bmm records a symbolic Bmm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bmm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bmm_out"
    assert out.dtype == DType.FLOAT32


def test_chain_matmul() -> None:
    """Verify torch_like.chain_matmul records a symbolic Chain_matmul_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.chain_matmul(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Chain_matmul_out"
    assert out.dtype == DType.FLOAT32


def test_cholesky() -> None:
    """Verify torch_like.cholesky records a symbolic Cholesky_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cholesky(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cholesky_out"
    assert out.dtype == DType.FLOAT32


def test_cholesky_inverse() -> None:
    """Verify torch_like.cholesky_inverse records a symbolic Cholesky_inverse_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cholesky_inverse(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cholesky_inverse_out"
    assert out.dtype == DType.FLOAT32


def test_cholesky_solve() -> None:
    """Verify torch_like.cholesky_solve records a symbolic Cholesky_solve_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cholesky_solve(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cholesky_solve_out"
    assert out.dtype == DType.FLOAT32


def test_corrcoef() -> None:
    """Verify torch_like.corrcoef records a symbolic Corrcoef_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.corrcoef(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Corrcoef_out"
    assert out.dtype == DType.FLOAT32


def test_cov() -> None:
    """Verify torch_like.cov records a symbolic Cov_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cov(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cov_out"
    assert out.dtype == DType.FLOAT32


def test_cross() -> None:
    """Verify torch_like.cross records a symbolic Cross_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cross(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cross_out"
    assert out.dtype == DType.FLOAT32


def test_diag() -> None:
    """Verify torch_like.diag records a symbolic Diag_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.diag(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Diag_out"
    assert out.dtype == DType.FLOAT32


def test_diag_embed() -> None:
    """Verify torch_like.diag_embed records a symbolic Diag_embed_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.diag_embed(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Diag_embed_out"
    assert out.dtype == DType.FLOAT32


def test_diagflat() -> None:
    """Verify torch_like.diagflat records a symbolic Diagflat_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.diagflat(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Diagflat_out"
    assert out.dtype == DType.FLOAT32


def test_diagonal() -> None:
    """Verify torch_like.diagonal records a symbolic Diagonal_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.diagonal(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Diagonal_out"
    assert out.dtype == DType.FLOAT32


def test_diagonal_copy() -> None:
    """Verify torch_like.diagonal_copy records a symbolic Diagonal_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.diagonal_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Diagonal_copy_out"
    assert out.dtype == DType.FLOAT32


def test_diagonal_scatter() -> None:
    """Verify torch_like.diagonal_scatter records a symbolic Diagonal_scatter_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.diagonal_scatter(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Diagonal_scatter_out"
    assert out.dtype == DType.FLOAT32


def test_dot() -> None:
    """Verify torch_like.dot records a symbolic Dot_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.dot(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Dot_out"
    assert out.dtype == DType.FLOAT32


def test_dsmm() -> None:
    """Verify torch_like.dsmm records a symbolic Dsmm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.dsmm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Dsmm_out"
    assert out.dtype == DType.FLOAT32


def test_frobenius_norm() -> None:
    """Verify torch_like.frobenius_norm records a symbolic Frobenius_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.frobenius_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Frobenius_norm_out"
    assert out.dtype == DType.FLOAT32


def test_geqrf() -> None:
    """Verify torch_like.geqrf records a symbolic Geqrf_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.geqrf(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Geqrf_out"
    assert out.dtype == DType.FLOAT32


def test_ger() -> None:
    """Verify torch_like.ger records a symbolic Ger_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ger(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ger_out"
    assert out.dtype == DType.FLOAT32


def test_hsmm() -> None:
    """Verify torch_like.hsmm records a symbolic Hsmm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.hsmm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Hsmm_out"
    assert out.dtype == DType.FLOAT32


def test_hspmm() -> None:
    """Verify torch_like.hspmm records a symbolic Hspmm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.hspmm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Hspmm_out"
    assert out.dtype == DType.FLOAT32


def test_inner() -> None:
    """Verify torch_like.inner records a symbolic Inner_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.inner(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Inner_out"
    assert out.dtype == DType.FLOAT32


def test_inverse() -> None:
    """Verify torch_like.inverse records a symbolic Inverse_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.inverse(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Inverse_out"
    assert out.dtype == DType.FLOAT32


def test_istft() -> None:
    """Verify torch_like.istft records a symbolic Istft_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.istft(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Istft_out"
    assert out.dtype == DType.FLOAT32


def test_kron() -> None:
    """Verify torch_like.kron records a symbolic Kron_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.kron(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Kron_out"
    assert out.dtype == DType.FLOAT32


def test_logdet() -> None:
    """Verify torch_like.logdet records a symbolic Logdet_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.logdet(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Logdet_out"
    assert out.dtype == DType.FLOAT32


def test_lu_solve() -> None:
    """Verify torch_like.lu_solve records a symbolic Lu_solve_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.lu_solve(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Lu_solve_out"
    assert out.dtype == DType.FLOAT32


def test_lu_unpack() -> None:
    """Verify torch_like.lu_unpack records a symbolic Lu_unpack_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.lu_unpack(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Lu_unpack_out"
    assert out.dtype == DType.FLOAT32


def test_matrix_exp() -> None:
    """Verify torch_like.matrix_exp records a symbolic Matrix_exp_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.matrix_exp(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Matrix_exp_out"
    assert out.dtype == DType.FLOAT32


def test_matrix_power() -> None:
    """Verify torch_like.matrix_power records a symbolic Matrix_power_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.matrix_power(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Matrix_power_out"
    assert out.dtype == DType.FLOAT32


def test_mm() -> None:
    """Verify torch_like.mm records a symbolic Mm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mm_out"
    assert out.dtype == DType.FLOAT32


def test_mv() -> None:
    """Verify torch_like.mv records a symbolic Mv_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mv(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mv_out"
    assert out.dtype == DType.FLOAT32


def test_norm() -> None:
    """Verify torch_like.norm records a symbolic Norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Norm_out"
    assert out.dtype == DType.FLOAT32


def test_nuclear_norm() -> None:
    """Verify torch_like.nuclear_norm records a symbolic Nuclear_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.nuclear_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Nuclear_norm_out"
    assert out.dtype == DType.FLOAT32


def test_orgqr() -> None:
    """Verify torch_like.orgqr records a symbolic Orgqr_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.orgqr(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Orgqr_out"
    assert out.dtype == DType.FLOAT32


def test_ormqr() -> None:
    """Verify torch_like.ormqr records a symbolic Ormqr_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ormqr(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ormqr_out"
    assert out.dtype == DType.FLOAT32


def test_outer() -> None:
    """Verify torch_like.outer records a symbolic Outer_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.outer(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Outer_out"
    assert out.dtype == DType.FLOAT32


def test_pinverse() -> None:
    """Verify torch_like.pinverse records a symbolic Pinverse_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.pinverse(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Pinverse_out"
    assert out.dtype == DType.FLOAT32


def test_qr() -> None:
    """Verify torch_like.qr records a symbolic Qr_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.qr(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Qr_out"
    assert out.dtype == DType.FLOAT32


def test_saddmm() -> None:
    """Verify torch_like.saddmm records a symbolic Saddmm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.saddmm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Saddmm_out"
    assert out.dtype == DType.FLOAT32


def test_slogdet() -> None:
    """Verify torch_like.slogdet records a symbolic Slogdet_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.slogdet(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Slogdet_out"
    assert out.dtype == DType.FLOAT32


def test_smm() -> None:
    """Verify torch_like.smm records a symbolic Smm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.smm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Smm_out"
    assert out.dtype == DType.FLOAT32


def test_spmm() -> None:
    """Verify torch_like.spmm records a symbolic Spmm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.spmm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Spmm_out"
    assert out.dtype == DType.FLOAT32


def test_sspaddmm() -> None:
    """Verify torch_like.sspaddmm records a symbolic Sspaddmm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sspaddmm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sspaddmm_out"
    assert out.dtype == DType.FLOAT32


def test_stft() -> None:
    """Verify torch_like.stft records a symbolic Stft_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.stft(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Stft_out"
    assert out.dtype == DType.FLOAT32


def test_svd() -> None:
    """Verify torch_like.svd records a symbolic Svd_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.svd(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Svd_out"
    assert out.dtype == DType.FLOAT32


def test_tensordot() -> None:
    """Verify torch_like.tensordot records a symbolic Tensordot_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.tensordot(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Tensordot_out"
    assert out.dtype == DType.FLOAT32


def test_triangular_solve() -> None:
    """Verify torch_like.triangular_solve records a symbolic Triangular_solve_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.triangular_solve(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Triangular_solve_out"
    assert out.dtype == DType.FLOAT32


def test_tril() -> None:
    """Verify torch_like.tril records a symbolic Tril_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.tril(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Tril_out"
    assert out.dtype == DType.FLOAT32


def test_triu() -> None:
    """Verify torch_like.triu records a symbolic Triu_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.triu(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Triu_out"
    assert out.dtype == DType.FLOAT32


def test_vdot() -> None:
    """Verify torch_like.vdot records a symbolic Vdot_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.vdot(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Vdot_out"
    assert out.dtype == DType.FLOAT32


def test_abs() -> None:
    """Verify torch_like.abs records a symbolic Abs_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.abs(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Abs_out"
    assert out.dtype == DType.FLOAT32


def test_acos() -> None:
    """Verify torch_like.acos records a symbolic Acos_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.acos(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Acos_out"
    assert out.dtype == DType.FLOAT32


def test_acosh() -> None:
    """Verify torch_like.acosh records a symbolic Acosh_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.acosh(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Acosh_out"
    assert out.dtype == DType.FLOAT32


def test_add() -> None:
    """Verify torch_like.add records a symbolic Add_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.add(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Add_out"
    assert out.dtype == DType.FLOAT32


def test_asin() -> None:
    """Verify torch_like.asin records a symbolic Asin_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.asin(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Asin_out"
    assert out.dtype == DType.FLOAT32


def test_asinh() -> None:
    """Verify torch_like.asinh records a symbolic Asinh_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.asinh(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Asinh_out"
    assert out.dtype == DType.FLOAT32


def test_atan() -> None:
    """Verify torch_like.atan records a symbolic Atan_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.atan(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Atan_out"
    assert out.dtype == DType.FLOAT32


def test_atanh() -> None:
    """Verify torch_like.atanh records a symbolic Atanh_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.atanh(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Atanh_out"
    assert out.dtype == DType.FLOAT32


def test_bitwise_and() -> None:
    """Verify torch_like.bitwise_and records a symbolic Bitwise_and_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bitwise_and(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bitwise_and_out"
    assert out.dtype == DType.FLOAT32


def test_bitwise_not() -> None:
    """Verify torch_like.bitwise_not records a symbolic Bitwise_not_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bitwise_not(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bitwise_not_out"
    assert out.dtype == DType.FLOAT32


def test_bitwise_or() -> None:
    """Verify torch_like.bitwise_or records a symbolic Bitwise_or_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bitwise_or(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bitwise_or_out"
    assert out.dtype == DType.FLOAT32


def test_bitwise_xor() -> None:
    """Verify torch_like.bitwise_xor records a symbolic Bitwise_xor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bitwise_xor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bitwise_xor_out"
    assert out.dtype == DType.FLOAT32


def test_ceil() -> None:
    """Verify torch_like.ceil records a symbolic Ceil_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ceil(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ceil_out"
    assert out.dtype == DType.FLOAT32


def test_cos() -> None:
    """Verify torch_like.cos records a symbolic Cos_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cos(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cos_out"
    assert out.dtype == DType.FLOAT32


def test_cosh() -> None:
    """Verify torch_like.cosh records a symbolic Cosh_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cosh(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cosh_out"
    assert out.dtype == DType.FLOAT32


def test_div() -> None:
    """Verify torch_like.div records a symbolic Div_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.div(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Div_out"
    assert out.dtype == DType.FLOAT32


def test_equal() -> None:
    """Verify torch_like.equal records a symbolic Equal_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.equal(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Equal_out"
    assert out.dtype == DType.FLOAT32


def test_erf() -> None:
    """Verify torch_like.erf records a symbolic Erf_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.erf(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Erf_out"
    assert out.dtype == DType.FLOAT32


def test_exp() -> None:
    """Verify torch_like.exp records a symbolic Exp_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.exp(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Exp_out"
    assert out.dtype == DType.FLOAT32


def test_floor() -> None:
    """Verify torch_like.floor records a symbolic Floor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.floor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Floor_out"
    assert out.dtype == DType.FLOAT32


def test_greater() -> None:
    """Verify torch_like.greater records a symbolic Greater_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.greater(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Greater_out"
    assert out.dtype == DType.FLOAT32


def test_isinf() -> None:
    """Verify torch_like.isinf records a symbolic Isinf_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.isinf(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Isinf_out"
    assert out.dtype == DType.FLOAT32


def test_isnan() -> None:
    """Verify torch_like.isnan records a symbolic Isnan_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.isnan(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Isnan_out"
    assert out.dtype == DType.FLOAT32


def test_less() -> None:
    """Verify torch_like.less records a symbolic Less_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.less(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Less_out"
    assert out.dtype == DType.FLOAT32


def test_mul() -> None:
    """Verify torch_like.mul records a symbolic Mul_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mul(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mul_out"
    assert out.dtype == DType.FLOAT32


def test_neg() -> None:
    """Verify torch_like.neg records a symbolic Neg_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.neg(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Neg_out"
    assert out.dtype == DType.FLOAT32


def test_pow() -> None:
    """Verify torch_like.pow records a symbolic Pow_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.pow(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Pow_out"
    assert out.dtype == DType.FLOAT32


def test_reciprocal() -> None:
    """Verify torch_like.reciprocal records a symbolic Reciprocal_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.reciprocal(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Reciprocal_out"
    assert out.dtype == DType.FLOAT32


def test_round() -> None:
    """Verify torch_like.round records a symbolic Round_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.round(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Round_out"
    assert out.dtype == DType.FLOAT32


def test_sign() -> None:
    """Verify torch_like.sign records a symbolic Sign_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sign(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sign_out"
    assert out.dtype == DType.FLOAT32


def test_sin() -> None:
    """Verify torch_like.sin records a symbolic Sin_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sin(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sin_out"
    assert out.dtype == DType.FLOAT32


def test_sinh() -> None:
    """Verify torch_like.sinh records a symbolic Sinh_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sinh(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sinh_out"
    assert out.dtype == DType.FLOAT32


def test_sub() -> None:
    """Verify torch_like.sub records a symbolic Sub_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sub(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sub_out"
    assert out.dtype == DType.FLOAT32


def test_tan() -> None:
    """Verify torch_like.tan records a symbolic Tan_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.tan(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Tan_out"
    assert out.dtype == DType.FLOAT32


def test_abs_() -> None:
    """Verify torch_like.abs_ records a symbolic Abs__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.abs_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Abs__out"
    assert out.dtype == DType.FLOAT32


def test_absolute() -> None:
    """Verify torch_like.absolute records a symbolic Absolute_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.absolute(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Absolute_out"
    assert out.dtype == DType.FLOAT32


def test_acos_() -> None:
    """Verify torch_like.acos_ records a symbolic Acos__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.acos_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Acos__out"
    assert out.dtype == DType.FLOAT32


def test_acosh_() -> None:
    """Verify torch_like.acosh_ records a symbolic Acosh__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.acosh_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Acosh__out"
    assert out.dtype == DType.FLOAT32


def test_addcdiv() -> None:
    """Verify torch_like.addcdiv records a symbolic Addcdiv_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.addcdiv(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Addcdiv_out"
    assert out.dtype == DType.FLOAT32


def test_addcmul() -> None:
    """Verify torch_like.addcmul records a symbolic Addcmul_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.addcmul(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Addcmul_out"
    assert out.dtype == DType.FLOAT32


def test_angle() -> None:
    """Verify torch_like.angle records a symbolic Angle_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.angle(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Angle_out"
    assert out.dtype == DType.FLOAT32


def test_arccos() -> None:
    """Verify torch_like.arccos records a symbolic Arccos_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arccos(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arccos_out"
    assert out.dtype == DType.FLOAT32


def test_arccos_() -> None:
    """Verify torch_like.arccos_ records a symbolic Arccos__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arccos_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arccos__out"
    assert out.dtype == DType.FLOAT32


def test_arccosh() -> None:
    """Verify torch_like.arccosh records a symbolic Arccosh_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arccosh(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arccosh_out"
    assert out.dtype == DType.FLOAT32


def test_arccosh_() -> None:
    """Verify torch_like.arccosh_ records a symbolic Arccosh__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arccosh_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arccosh__out"
    assert out.dtype == DType.FLOAT32


def test_arcsin() -> None:
    """Verify torch_like.arcsin records a symbolic Arcsin_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arcsin(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arcsin_out"
    assert out.dtype == DType.FLOAT32


def test_arcsin_() -> None:
    """Verify torch_like.arcsin_ records a symbolic Arcsin__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arcsin_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arcsin__out"
    assert out.dtype == DType.FLOAT32


def test_arcsinh() -> None:
    """Verify torch_like.arcsinh records a symbolic Arcsinh_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arcsinh(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arcsinh_out"
    assert out.dtype == DType.FLOAT32


def test_arcsinh_() -> None:
    """Verify torch_like.arcsinh_ records a symbolic Arcsinh__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arcsinh_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arcsinh__out"
    assert out.dtype == DType.FLOAT32


def test_arctan() -> None:
    """Verify torch_like.arctan records a symbolic Arctan_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arctan(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arctan_out"
    assert out.dtype == DType.FLOAT32


def test_arctan2() -> None:
    """Verify torch_like.arctan2 records a symbolic Arctan2_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arctan2(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arctan2_out"
    assert out.dtype == DType.FLOAT32


def test_arctan_() -> None:
    """Verify torch_like.arctan_ records a symbolic Arctan__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arctan_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arctan__out"
    assert out.dtype == DType.FLOAT32


def test_arctanh() -> None:
    """Verify torch_like.arctanh records a symbolic Arctanh_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arctanh(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arctanh_out"
    assert out.dtype == DType.FLOAT32


def test_arctanh_() -> None:
    """Verify torch_like.arctanh_ records a symbolic Arctanh__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.arctanh_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Arctanh__out"
    assert out.dtype == DType.FLOAT32


def test_asin_() -> None:
    """Verify torch_like.asin_ records a symbolic Asin__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.asin_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Asin__out"
    assert out.dtype == DType.FLOAT32


def test_asinh_() -> None:
    """Verify torch_like.asinh_ records a symbolic Asinh__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.asinh_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Asinh__out"
    assert out.dtype == DType.FLOAT32


def test_atan2() -> None:
    """Verify torch_like.atan2 records a symbolic Atan2_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.atan2(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Atan2_out"
    assert out.dtype == DType.FLOAT32


def test_atan_() -> None:
    """Verify torch_like.atan_ records a symbolic Atan__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.atan_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Atan__out"
    assert out.dtype == DType.FLOAT32


def test_atanh_() -> None:
    """Verify torch_like.atanh_ records a symbolic Atanh__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.atanh_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Atanh__out"
    assert out.dtype == DType.FLOAT32


def test_bitwise_left_shift() -> None:
    """Verify torch_like.bitwise_left_shift records a symbolic Bitwise_left_shift_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bitwise_left_shift(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bitwise_left_shift_out"
    assert out.dtype == DType.FLOAT32


def test_bitwise_right_shift() -> None:
    """Verify torch_like.bitwise_right_shift records a symbolic Bitwise_right_shift_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bitwise_right_shift(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bitwise_right_shift_out"
    assert out.dtype == DType.FLOAT32


def test_cdist() -> None:
    """Verify torch_like.cdist records a symbolic Cdist_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cdist(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cdist_out"
    assert out.dtype == DType.FLOAT32


def test_ceil_() -> None:
    """Verify torch_like.ceil_ records a symbolic Ceil__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ceil_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ceil__out"
    assert out.dtype == DType.FLOAT32


def test_clamp() -> None:
    """Verify torch_like.clamp records a symbolic Clamp_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.clamp(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Clamp_out"
    assert out.dtype == DType.FLOAT32


def test_clamp_() -> None:
    """Verify torch_like.clamp_ records a symbolic Clamp__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.clamp_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Clamp__out"
    assert out.dtype == DType.FLOAT32


def test_clamp_max() -> None:
    """Verify torch_like.clamp_max records a symbolic Clamp_max_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.clamp_max(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Clamp_max_out"
    assert out.dtype == DType.FLOAT32


def test_clamp_max_() -> None:
    """Verify torch_like.clamp_max_ records a symbolic Clamp_max__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.clamp_max_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Clamp_max__out"
    assert out.dtype == DType.FLOAT32


def test_clamp_min() -> None:
    """Verify torch_like.clamp_min records a symbolic Clamp_min_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.clamp_min(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Clamp_min_out"
    assert out.dtype == DType.FLOAT32


def test_clamp_min_() -> None:
    """Verify torch_like.clamp_min_ records a symbolic Clamp_min__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.clamp_min_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Clamp_min__out"
    assert out.dtype == DType.FLOAT32


def test_clip_() -> None:
    """Verify torch_like.clip_ records a symbolic Clip__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.clip_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Clip__out"
    assert out.dtype == DType.FLOAT32


def test_complex() -> None:
    """Verify torch_like.complex records a symbolic Complex_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.complex(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Complex_out"
    assert out.dtype == DType.FLOAT32


def test_conj() -> None:
    """Verify torch_like.conj records a symbolic Conj_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.conj(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Conj_out"
    assert out.dtype == DType.FLOAT32


def test_conj_physical() -> None:
    """Verify torch_like.conj_physical records a symbolic Conj_physical_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.conj_physical(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Conj_physical_out"
    assert out.dtype == DType.FLOAT32


def test_conj_physical_() -> None:
    """Verify torch_like.conj_physical_ records a symbolic Conj_physical__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.conj_physical_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Conj_physical__out"
    assert out.dtype == DType.FLOAT32


def test_copysign() -> None:
    """Verify torch_like.copysign records a symbolic Copysign_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.copysign(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Copysign_out"
    assert out.dtype == DType.FLOAT32


def test_cos_() -> None:
    """Verify torch_like.cos_ records a symbolic Cos__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cos_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cos__out"
    assert out.dtype == DType.FLOAT32


def test_cosh_() -> None:
    """Verify torch_like.cosh_ records a symbolic Cosh__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cosh_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cosh__out"
    assert out.dtype == DType.FLOAT32


def test_deg2rad() -> None:
    """Verify torch_like.deg2rad records a symbolic Deg2rad_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.deg2rad(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Deg2rad_out"
    assert out.dtype == DType.FLOAT32


def test_deg2rad_() -> None:
    """Verify torch_like.deg2rad_ records a symbolic Deg2rad__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.deg2rad_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Deg2rad__out"
    assert out.dtype == DType.FLOAT32


def test_diff() -> None:
    """Verify torch_like.diff records a symbolic Diff_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.diff(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Diff_out"
    assert out.dtype == DType.FLOAT32


def test_digamma() -> None:
    """Verify torch_like.digamma records a symbolic Digamma_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.digamma(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Digamma_out"
    assert out.dtype == DType.FLOAT32


def test_divide() -> None:
    """Verify torch_like.divide records a symbolic Divide_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.divide(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Divide_out"
    assert out.dtype == DType.FLOAT32


def test_eq() -> None:
    """Verify torch_like.eq records a symbolic Eq_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.eq(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Eq_out"
    assert out.dtype == DType.FLOAT32


def test_erf_() -> None:
    """Verify torch_like.erf_ records a symbolic Erf__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.erf_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Erf__out"
    assert out.dtype == DType.FLOAT32


def test_erfc() -> None:
    """Verify torch_like.erfc records a symbolic Erfc_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.erfc(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Erfc_out"
    assert out.dtype == DType.FLOAT32


def test_erfc_() -> None:
    """Verify torch_like.erfc_ records a symbolic Erfc__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.erfc_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Erfc__out"
    assert out.dtype == DType.FLOAT32


def test_erfinv() -> None:
    """Verify torch_like.erfinv records a symbolic Erfinv_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.erfinv(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Erfinv_out"
    assert out.dtype == DType.FLOAT32


def test_exp2() -> None:
    """Verify torch_like.exp2 records a symbolic Exp2_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.exp2(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Exp2_out"
    assert out.dtype == DType.FLOAT32


def test_exp2_() -> None:
    """Verify torch_like.exp2_ records a symbolic Exp2__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.exp2_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Exp2__out"
    assert out.dtype == DType.FLOAT32


def test_exp_() -> None:
    """Verify torch_like.exp_ records a symbolic Exp__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.exp_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Exp__out"
    assert out.dtype == DType.FLOAT32


def test_expm1_() -> None:
    """Verify torch_like.expm1_ records a symbolic Expm1__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.expm1_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Expm1__out"
    assert out.dtype == DType.FLOAT32


def test_fix() -> None:
    """Verify torch_like.fix records a symbolic Fix_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fix(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fix_out"
    assert out.dtype == DType.FLOAT32


def test_fix_() -> None:
    """Verify torch_like.fix_ records a symbolic Fix__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fix_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fix__out"
    assert out.dtype == DType.FLOAT32


def test_float_power() -> None:
    """Verify torch_like.float_power records a symbolic Float_power_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.float_power(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Float_power_out"
    assert out.dtype == DType.FLOAT32


def test_floor_() -> None:
    """Verify torch_like.floor_ records a symbolic Floor__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.floor_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Floor__out"
    assert out.dtype == DType.FLOAT32


def test_floor_divide() -> None:
    """Verify torch_like.floor_divide records a symbolic Floor_divide_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.floor_divide(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Floor_divide_out"
    assert out.dtype == DType.FLOAT32


def test_fmax() -> None:
    """Verify torch_like.fmax records a symbolic Fmax_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fmax(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fmax_out"
    assert out.dtype == DType.FLOAT32


def test_fmin() -> None:
    """Verify torch_like.fmin records a symbolic Fmin_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fmin(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fmin_out"
    assert out.dtype == DType.FLOAT32


def test_frac() -> None:
    """Verify torch_like.frac records a symbolic Frac_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.frac(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Frac_out"
    assert out.dtype == DType.FLOAT32


def test_frac_() -> None:
    """Verify torch_like.frac_ records a symbolic Frac__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.frac_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Frac__out"
    assert out.dtype == DType.FLOAT32


def test_frexp() -> None:
    """Verify torch_like.frexp records a symbolic Frexp_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.frexp(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Frexp_out"
    assert out.dtype == DType.FLOAT32


def test_gcd() -> None:
    """Verify torch_like.gcd records a symbolic Gcd_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.gcd(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Gcd_out"
    assert out.dtype == DType.FLOAT32


def test_gcd_() -> None:
    """Verify torch_like.gcd_ records a symbolic Gcd__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.gcd_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Gcd__out"
    assert out.dtype == DType.FLOAT32


def test_ge() -> None:
    """Verify torch_like.ge records a symbolic Ge_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ge(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ge_out"
    assert out.dtype == DType.FLOAT32


def test_gradient() -> None:
    """Verify torch_like.gradient records a symbolic Gradient_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.gradient(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Gradient_out"
    assert out.dtype == DType.FLOAT32


def test_greater_equal() -> None:
    """Verify torch_like.greater_equal records a symbolic Greater_equal_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.greater_equal(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Greater_equal_out"
    assert out.dtype == DType.FLOAT32


def test_gt() -> None:
    """Verify torch_like.gt records a symbolic Gt_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.gt(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Gt_out"
    assert out.dtype == DType.FLOAT32


def test_hash_tensor() -> None:
    """Verify torch_like.hash_tensor records a symbolic Hash_tensor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.hash_tensor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Hash_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_heaviside() -> None:
    """Verify torch_like.heaviside records a symbolic Heaviside_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.heaviside(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Heaviside_out"
    assert out.dtype == DType.FLOAT32


def test_hypot() -> None:
    """Verify torch_like.hypot records a symbolic Hypot_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.hypot(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Hypot_out"
    assert out.dtype == DType.FLOAT32


def test_i0() -> None:
    """Verify torch_like.i0 records a symbolic I0_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.i0(x)
    assert isinstance(out, IRTensor)
    assert out.name == "I0_out"
    assert out.dtype == DType.FLOAT32


def test_i0_() -> None:
    """Verify torch_like.i0_ records a symbolic I0__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.i0_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "I0__out"
    assert out.dtype == DType.FLOAT32


def test_igamma() -> None:
    """Verify torch_like.igamma records a symbolic Igamma_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.igamma(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Igamma_out"
    assert out.dtype == DType.FLOAT32


def test_igammac() -> None:
    """Verify torch_like.igammac records a symbolic Igammac_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.igammac(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Igammac_out"
    assert out.dtype == DType.FLOAT32


def test_imag() -> None:
    """Verify torch_like.imag records a symbolic Imag_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.imag(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Imag_out"
    assert out.dtype == DType.FLOAT32


def test_is_complex() -> None:
    """Verify torch_like.is_complex records a symbolic Is_complex_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.is_complex(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Is_complex_out"
    assert out.dtype == DType.FLOAT32


def test_is_conj() -> None:
    """Verify torch_like.is_conj records a symbolic Is_conj_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.is_conj(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Is_conj_out"
    assert out.dtype == DType.FLOAT32


def test_is_floating_point() -> None:
    """Verify torch_like.is_floating_point records a symbolic Is_floating_point_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.is_floating_point(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Is_floating_point_out"
    assert out.dtype == DType.FLOAT32


def test_is_neg() -> None:
    """Verify torch_like.is_neg records a symbolic Is_neg_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.is_neg(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Is_neg_out"
    assert out.dtype == DType.FLOAT32


def test_is_nonzero() -> None:
    """Verify torch_like.is_nonzero records a symbolic Is_nonzero_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.is_nonzero(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Is_nonzero_out"
    assert out.dtype == DType.FLOAT32


def test_is_same_size() -> None:
    """Verify torch_like.is_same_size records a symbolic Is_same_size_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.is_same_size(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Is_same_size_out"
    assert out.dtype == DType.FLOAT32


def test_is_signed() -> None:
    """Verify torch_like.is_signed records a symbolic Is_signed_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.is_signed(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Is_signed_out"
    assert out.dtype == DType.FLOAT32


def test_isclose() -> None:
    """Verify torch_like.isclose records a symbolic Isclose_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.isclose(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Isclose_out"
    assert out.dtype == DType.FLOAT32


def test_isneginf() -> None:
    """Verify torch_like.isneginf records a symbolic Isneginf_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.isneginf(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Isneginf_out"
    assert out.dtype == DType.FLOAT32


def test_isposinf() -> None:
    """Verify torch_like.isposinf records a symbolic Isposinf_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.isposinf(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Isposinf_out"
    assert out.dtype == DType.FLOAT32


def test_isreal() -> None:
    """Verify torch_like.isreal records a symbolic Isreal_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.isreal(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Isreal_out"
    assert out.dtype == DType.FLOAT32


def test_lcm() -> None:
    """Verify torch_like.lcm records a symbolic Lcm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.lcm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Lcm_out"
    assert out.dtype == DType.FLOAT32


def test_lcm_() -> None:
    """Verify torch_like.lcm_ records a symbolic Lcm__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.lcm_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Lcm__out"
    assert out.dtype == DType.FLOAT32


def test_ldexp() -> None:
    """Verify torch_like.ldexp records a symbolic Ldexp_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ldexp(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ldexp_out"
    assert out.dtype == DType.FLOAT32


def test_ldexp_() -> None:
    """Verify torch_like.ldexp_ records a symbolic Ldexp__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ldexp_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ldexp__out"
    assert out.dtype == DType.FLOAT32


def test_le() -> None:
    """Verify torch_like.le records a symbolic Le_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.le(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Le_out"
    assert out.dtype == DType.FLOAT32


def test_lerp() -> None:
    """Verify torch_like.lerp records a symbolic Lerp_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.lerp(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Lerp_out"
    assert out.dtype == DType.FLOAT32


def test_less_equal() -> None:
    """Verify torch_like.less_equal records a symbolic Less_equal_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.less_equal(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Less_equal_out"
    assert out.dtype == DType.FLOAT32


def test_lgamma() -> None:
    """Verify torch_like.lgamma records a symbolic Lgamma_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.lgamma(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Lgamma_out"
    assert out.dtype == DType.FLOAT32


def test_log10_() -> None:
    """Verify torch_like.log10_ records a symbolic Log10__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.log10_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Log10__out"
    assert out.dtype == DType.FLOAT32


def test_log1p_() -> None:
    """Verify torch_like.log1p_ records a symbolic Log1p__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.log1p_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Log1p__out"
    assert out.dtype == DType.FLOAT32


def test_log2_() -> None:
    """Verify torch_like.log2_ records a symbolic Log2__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.log2_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Log2__out"
    assert out.dtype == DType.FLOAT32


def test_log_() -> None:
    """Verify torch_like.log_ records a symbolic Log__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.log_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Log__out"
    assert out.dtype == DType.FLOAT32


def test_logaddexp() -> None:
    """Verify torch_like.logaddexp records a symbolic Logaddexp_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.logaddexp(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Logaddexp_out"
    assert out.dtype == DType.FLOAT32


def test_logaddexp2() -> None:
    """Verify torch_like.logaddexp2 records a symbolic Logaddexp2_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.logaddexp2(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Logaddexp2_out"
    assert out.dtype == DType.FLOAT32


def test_logit() -> None:
    """Verify torch_like.logit records a symbolic Logit_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.logit(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Logit_out"
    assert out.dtype == DType.FLOAT32


def test_logit_() -> None:
    """Verify torch_like.logit_ records a symbolic Logit__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.logit_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Logit__out"
    assert out.dtype == DType.FLOAT32


def test_lt() -> None:
    """Verify torch_like.lt records a symbolic Lt_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.lt(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Lt_out"
    assert out.dtype == DType.FLOAT32


def test_multiply() -> None:
    """Verify torch_like.multiply records a symbolic Multiply_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.multiply(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Multiply_out"
    assert out.dtype == DType.FLOAT32


def test_mvlgamma() -> None:
    """Verify torch_like.mvlgamma records a symbolic Mvlgamma_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mvlgamma(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mvlgamma_out"
    assert out.dtype == DType.FLOAT32


def test_nan_to_num() -> None:
    """Verify torch_like.nan_to_num records a symbolic Nan_to_num_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.nan_to_num(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Nan_to_num_out"
    assert out.dtype == DType.FLOAT32


def test_nan_to_num_() -> None:
    """Verify torch_like.nan_to_num_ records a symbolic Nan_to_num__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.nan_to_num_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Nan_to_num__out"
    assert out.dtype == DType.FLOAT32


def test_ne() -> None:
    """Verify torch_like.ne records a symbolic Ne_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ne(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ne_out"
    assert out.dtype == DType.FLOAT32


def test_neg_() -> None:
    """Verify torch_like.neg_ records a symbolic Neg__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.neg_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Neg__out"
    assert out.dtype == DType.FLOAT32


def test_negative() -> None:
    """Verify torch_like.negative records a symbolic Negative_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.negative(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Negative_out"
    assert out.dtype == DType.FLOAT32


def test_negative_() -> None:
    """Verify torch_like.negative_ records a symbolic Negative__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.negative_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Negative__out"
    assert out.dtype == DType.FLOAT32


def test_nextafter() -> None:
    """Verify torch_like.nextafter records a symbolic Nextafter_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.nextafter(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Nextafter_out"
    assert out.dtype == DType.FLOAT32


def test_not_equal() -> None:
    """Verify torch_like.not_equal records a symbolic Not_equal_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.not_equal(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Not_equal_out"
    assert out.dtype == DType.FLOAT32


def test_polar() -> None:
    """Verify torch_like.polar records a symbolic Polar_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.polar(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Polar_out"
    assert out.dtype == DType.FLOAT32


def test_polygamma() -> None:
    """Verify torch_like.polygamma records a symbolic Polygamma_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.polygamma(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Polygamma_out"
    assert out.dtype == DType.FLOAT32


def test_positive() -> None:
    """Verify torch_like.positive records a symbolic Positive_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.positive(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Positive_out"
    assert out.dtype == DType.FLOAT32


def test_rad2deg() -> None:
    """Verify torch_like.rad2deg records a symbolic Rad2deg_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rad2deg(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rad2deg_out"
    assert out.dtype == DType.FLOAT32


def test_rad2deg_() -> None:
    """Verify torch_like.rad2deg_ records a symbolic Rad2deg__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rad2deg_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rad2deg__out"
    assert out.dtype == DType.FLOAT32


def test_real() -> None:
    """Verify torch_like.real records a symbolic Real_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.real(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Real_out"
    assert out.dtype == DType.FLOAT32


def test_reciprocal_() -> None:
    """Verify torch_like.reciprocal_ records a symbolic Reciprocal__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.reciprocal_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Reciprocal__out"
    assert out.dtype == DType.FLOAT32


def test_remainder() -> None:
    """Verify torch_like.remainder records a symbolic Remainder_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.remainder(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Remainder_out"
    assert out.dtype == DType.FLOAT32


def test_renorm() -> None:
    """Verify torch_like.renorm records a symbolic Renorm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.renorm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Renorm_out"
    assert out.dtype == DType.FLOAT32


def test_resolve_conj() -> None:
    """Verify torch_like.resolve_conj records a symbolic Resolve_conj_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.resolve_conj(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Resolve_conj_out"
    assert out.dtype == DType.FLOAT32


def test_resolve_neg() -> None:
    """Verify torch_like.resolve_neg records a symbolic Resolve_neg_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.resolve_neg(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Resolve_neg_out"
    assert out.dtype == DType.FLOAT32


def test_round_() -> None:
    """Verify torch_like.round_ records a symbolic Round__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.round_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Round__out"
    assert out.dtype == DType.FLOAT32


def test_rsqrt() -> None:
    """Verify torch_like.rsqrt records a symbolic Rsqrt_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rsqrt(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rsqrt_out"
    assert out.dtype == DType.FLOAT32


def test_rsqrt_() -> None:
    """Verify torch_like.rsqrt_ records a symbolic Rsqrt__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rsqrt_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rsqrt__out"
    assert out.dtype == DType.FLOAT32


def test_rsub() -> None:
    """Verify torch_like.rsub records a symbolic Rsub_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rsub(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rsub_out"
    assert out.dtype == DType.FLOAT32


def test_sgn() -> None:
    """Verify torch_like.sgn records a symbolic Sgn_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sgn(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sgn_out"
    assert out.dtype == DType.FLOAT32


def test_signbit() -> None:
    """Verify torch_like.signbit records a symbolic Signbit_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.signbit(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Signbit_out"
    assert out.dtype == DType.FLOAT32


def test_sin_() -> None:
    """Verify torch_like.sin_ records a symbolic Sin__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sin_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sin__out"
    assert out.dtype == DType.FLOAT32


def test_sinc() -> None:
    """Verify torch_like.sinc records a symbolic Sinc_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sinc(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sinc_out"
    assert out.dtype == DType.FLOAT32


def test_sinc_() -> None:
    """Verify torch_like.sinc_ records a symbolic Sinc__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sinc_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sinc__out"
    assert out.dtype == DType.FLOAT32


def test_sinh_() -> None:
    """Verify torch_like.sinh_ records a symbolic Sinh__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sinh_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sinh__out"
    assert out.dtype == DType.FLOAT32


def test_sqrt_() -> None:
    """Verify torch_like.sqrt_ records a symbolic Sqrt__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sqrt_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sqrt__out"
    assert out.dtype == DType.FLOAT32


def test_square() -> None:
    """Verify torch_like.square records a symbolic Square_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.square(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Square_out"
    assert out.dtype == DType.FLOAT32


def test_square_() -> None:
    """Verify torch_like.square_ records a symbolic Square__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.square_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Square__out"
    assert out.dtype == DType.FLOAT32


def test_subtract() -> None:
    """Verify torch_like.subtract records a symbolic Subtract_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.subtract(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Subtract_out"
    assert out.dtype == DType.FLOAT32


def test_tan_() -> None:
    """Verify torch_like.tan_ records a symbolic Tan__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.tan_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Tan__out"
    assert out.dtype == DType.FLOAT32


def test_true_divide() -> None:
    """Verify torch_like.true_divide records a symbolic True_divide_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.true_divide(x)
    assert isinstance(out, IRTensor)
    assert out.name == "True_divide_out"
    assert out.dtype == DType.FLOAT32


def test_trunc() -> None:
    """Verify torch_like.trunc records a symbolic Trunc_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.trunc(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Trunc_out"
    assert out.dtype == DType.FLOAT32


def test_trunc_() -> None:
    """Verify torch_like.trunc_ records a symbolic Trunc__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.trunc_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Trunc__out"
    assert out.dtype == DType.FLOAT32


def test_xlogy() -> None:
    """Verify torch_like.xlogy records a symbolic Xlogy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.xlogy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Xlogy_out"
    assert out.dtype == DType.FLOAT32


def test_xlogy_() -> None:
    """Verify torch_like.xlogy_ records a symbolic Xlogy__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.xlogy_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Xlogy__out"
    assert out.dtype == DType.FLOAT32


def test_adaptive_avg_pool1d() -> None:
    """Verify torch_like.adaptive_avg_pool1d records a symbolic Adaptive_avg_pool1d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.adaptive_avg_pool1d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Adaptive_avg_pool1d_out"
    assert out.dtype == DType.FLOAT32


def test_adaptive_max_pool1d() -> None:
    """Verify torch_like.adaptive_max_pool1d records a symbolic Adaptive_max_pool1d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.adaptive_max_pool1d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Adaptive_max_pool1d_out"
    assert out.dtype == DType.FLOAT32


def test_affine_grid_generator() -> None:
    """Verify torch_like.affine_grid_generator records a symbolic Affine_grid_generator_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.affine_grid_generator(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Affine_grid_generator_out"
    assert out.dtype == DType.FLOAT32


def test_alpha_dropout() -> None:
    """Verify torch_like.alpha_dropout records a symbolic Alpha_dropout_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.alpha_dropout(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Alpha_dropout_out"
    assert out.dtype == DType.FLOAT32


def test_alpha_dropout_() -> None:
    """Verify torch_like.alpha_dropout_ records a symbolic Alpha_dropout__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.alpha_dropout_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Alpha_dropout__out"
    assert out.dtype == DType.FLOAT32


def test_avg_pool1d() -> None:
    """Verify torch_like.avg_pool1d records a symbolic Avg_pool1d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.avg_pool1d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Avg_pool1d_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm() -> None:
    """Verify torch_like.batch_norm records a symbolic Batch_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.batch_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Batch_norm_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_backward_elemt() -> None:
    """Verify torch_like.batch_norm_backward_elemt records a symbolic Batch_norm_backward_elemt_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.batch_norm_backward_elemt(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Batch_norm_backward_elemt_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_backward_reduce() -> None:
    """Verify torch_like.batch_norm_backward_reduce records a symbolic Batch_norm_backward_reduce_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.batch_norm_backward_reduce(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Batch_norm_backward_reduce_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_elemt() -> None:
    """Verify torch_like.batch_norm_elemt records a symbolic Batch_norm_elemt_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.batch_norm_elemt(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Batch_norm_elemt_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_gather_stats() -> None:
    """Verify torch_like.batch_norm_gather_stats records a symbolic Batch_norm_gather_stats_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.batch_norm_gather_stats(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Batch_norm_gather_stats_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_gather_stats_with_counts() -> None:
    """Verify torch_like.batch_norm_gather_stats_with_counts records a symbolic Batch_norm_gather_stats_with_counts_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.batch_norm_gather_stats_with_counts(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Batch_norm_gather_stats_with_counts_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_stats() -> None:
    """Verify torch_like.batch_norm_stats records a symbolic Batch_norm_stats_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.batch_norm_stats(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Batch_norm_stats_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_update_stats() -> None:
    """Verify torch_like.batch_norm_update_stats records a symbolic Batch_norm_update_stats_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.batch_norm_update_stats(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Batch_norm_update_stats_out"
    assert out.dtype == DType.FLOAT32


def test_bilinear() -> None:
    """Verify torch_like.bilinear records a symbolic Bilinear_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bilinear(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bilinear_out"
    assert out.dtype == DType.FLOAT32


def test_binary_cross_entropy_with_logits() -> None:
    """Verify torch_like.binary_cross_entropy_with_logits records a symbolic Binary_cross_entropy_with_logits_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.binary_cross_entropy_with_logits(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Binary_cross_entropy_with_logits_out"
    assert out.dtype == DType.FLOAT32


def test_celu_() -> None:
    """Verify torch_like.celu_ records a symbolic Celu__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.celu_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Celu__out"
    assert out.dtype == DType.FLOAT32


def test_channel_shuffle() -> None:
    """Verify torch_like.channel_shuffle records a symbolic Channel_shuffle_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.channel_shuffle(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Channel_shuffle_out"
    assert out.dtype == DType.FLOAT32


def test_choose_qparams_optimized() -> None:
    """Verify torch_like.choose_qparams_optimized records a symbolic Choose_qparams_optimized_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.choose_qparams_optimized(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Choose_qparams_optimized_out"
    assert out.dtype == DType.FLOAT32


def test_constant_pad_nd() -> None:
    """Verify torch_like.constant_pad_nd records a symbolic Constant_pad_nd_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.constant_pad_nd(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Constant_pad_nd_out"
    assert out.dtype == DType.FLOAT32


def test_conv1d() -> None:
    """Verify torch_like.conv1d records a symbolic Conv1d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.conv1d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Conv1d_out"
    assert out.dtype == DType.FLOAT32


def test_conv2d() -> None:
    """Verify torch_like.conv2d records a symbolic Conv2d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.conv2d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Conv2d_out"
    assert out.dtype == DType.FLOAT32


def test_conv3d() -> None:
    """Verify torch_like.conv3d records a symbolic Conv3d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.conv3d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Conv3d_out"
    assert out.dtype == DType.FLOAT32


def test_conv_tbc() -> None:
    """Verify torch_like.conv_tbc records a symbolic Conv_tbc_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.conv_tbc(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Conv_tbc_out"
    assert out.dtype == DType.FLOAT32


def test_conv_transpose1d() -> None:
    """Verify torch_like.conv_transpose1d records a symbolic Conv_transpose1d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.conv_transpose1d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Conv_transpose1d_out"
    assert out.dtype == DType.FLOAT32


def test_conv_transpose2d() -> None:
    """Verify torch_like.conv_transpose2d records a symbolic Conv_transpose2d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.conv_transpose2d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Conv_transpose2d_out"
    assert out.dtype == DType.FLOAT32


def test_conv_transpose3d() -> None:
    """Verify torch_like.conv_transpose3d records a symbolic Conv_transpose3d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.conv_transpose3d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Conv_transpose3d_out"
    assert out.dtype == DType.FLOAT32


def test_convolution() -> None:
    """Verify torch_like.convolution records a symbolic Convolution_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.convolution(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Convolution_out"
    assert out.dtype == DType.FLOAT32


def test_cosine_embedding_loss() -> None:
    """Verify torch_like.cosine_embedding_loss records a symbolic Cosine_embedding_loss_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cosine_embedding_loss(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cosine_embedding_loss_out"
    assert out.dtype == DType.FLOAT32


def test_cosine_similarity() -> None:
    """Verify torch_like.cosine_similarity records a symbolic Cosine_similarity_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cosine_similarity(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cosine_similarity_out"
    assert out.dtype == DType.FLOAT32


def test_ctc_loss() -> None:
    """Verify torch_like.ctc_loss records a symbolic Ctc_loss_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ctc_loss(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ctc_loss_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_affine_grid_generator() -> None:
    """Verify torch_like.cudnn_affine_grid_generator records a symbolic Cudnn_affine_grid_generator_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cudnn_affine_grid_generator(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cudnn_affine_grid_generator_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_batch_norm() -> None:
    """Verify torch_like.cudnn_batch_norm records a symbolic Cudnn_batch_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cudnn_batch_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cudnn_batch_norm_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_convolution() -> None:
    """Verify torch_like.cudnn_convolution records a symbolic Cudnn_convolution_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cudnn_convolution(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cudnn_convolution_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_convolution_add_relu() -> None:
    """Verify torch_like.cudnn_convolution_add_relu records a symbolic Cudnn_convolution_add_relu_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cudnn_convolution_add_relu(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cudnn_convolution_add_relu_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_convolution_relu() -> None:
    """Verify torch_like.cudnn_convolution_relu records a symbolic Cudnn_convolution_relu_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cudnn_convolution_relu(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cudnn_convolution_relu_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_convolution_transpose() -> None:
    """Verify torch_like.cudnn_convolution_transpose records a symbolic Cudnn_convolution_transpose_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cudnn_convolution_transpose(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cudnn_convolution_transpose_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_grid_sampler() -> None:
    """Verify torch_like.cudnn_grid_sampler records a symbolic Cudnn_grid_sampler_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cudnn_grid_sampler(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cudnn_grid_sampler_out"
    assert out.dtype == DType.FLOAT32


def test_dequantize() -> None:
    """Verify torch_like.dequantize records a symbolic Dequantize_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.dequantize(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Dequantize_out"
    assert out.dtype == DType.FLOAT32


def test_dropout_() -> None:
    """Verify torch_like.dropout_ records a symbolic Dropout__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.dropout_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Dropout__out"
    assert out.dtype == DType.FLOAT32


def test_embedding() -> None:
    """Verify torch_like.embedding records a symbolic Embedding_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.embedding(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Embedding_out"
    assert out.dtype == DType.FLOAT32


def test_embedding_bag() -> None:
    """Verify torch_like.embedding_bag records a symbolic Embedding_bag_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.embedding_bag(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Embedding_bag_out"
    assert out.dtype == DType.FLOAT32


def test_embedding_renorm_() -> None:
    """Verify torch_like.embedding_renorm_ records a symbolic Embedding_renorm__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.embedding_renorm_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Embedding_renorm__out"
    assert out.dtype == DType.FLOAT32


def test_fake_quantize_per_channel_affine() -> None:
    """Verify torch_like.fake_quantize_per_channel_affine records a symbolic Fake_quantize_per_channel_affine_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fake_quantize_per_channel_affine(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fake_quantize_per_channel_affine_out"
    assert out.dtype == DType.FLOAT32


def test_fake_quantize_per_tensor_affine() -> None:
    """Verify torch_like.fake_quantize_per_tensor_affine records a symbolic Fake_quantize_per_tensor_affine_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fake_quantize_per_tensor_affine(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fake_quantize_per_tensor_affine_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_linear_fp16_weight() -> None:
    """Verify torch_like.fbgemm_linear_fp16_weight records a symbolic Fbgemm_linear_fp16_weight_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fbgemm_linear_fp16_weight(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fbgemm_linear_fp16_weight_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_linear_fp16_weight_fp32_activation() -> None:
    """Verify torch_like.fbgemm_linear_fp16_weight_fp32_activation records a symbolic Fbgemm_linear_fp16_weight_fp32_activation_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fbgemm_linear_fp16_weight_fp32_activation(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fbgemm_linear_fp16_weight_fp32_activation_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_linear_int8_weight() -> None:
    """Verify torch_like.fbgemm_linear_int8_weight records a symbolic Fbgemm_linear_int8_weight_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fbgemm_linear_int8_weight(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fbgemm_linear_int8_weight_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_linear_int8_weight_fp32_activation() -> None:
    """Verify torch_like.fbgemm_linear_int8_weight_fp32_activation records a symbolic Fbgemm_linear_int8_weight_fp32_activation_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fbgemm_linear_int8_weight_fp32_activation(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fbgemm_linear_int8_weight_fp32_activation_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_linear_quantize_weight() -> None:
    """Verify torch_like.fbgemm_linear_quantize_weight records a symbolic Fbgemm_linear_quantize_weight_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fbgemm_linear_quantize_weight(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fbgemm_linear_quantize_weight_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_pack_gemm_matrix_fp16() -> None:
    """Verify torch_like.fbgemm_pack_gemm_matrix_fp16 records a symbolic Fbgemm_pack_gemm_matrix_fp16_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fbgemm_pack_gemm_matrix_fp16(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fbgemm_pack_gemm_matrix_fp16_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_pack_quantized_matrix() -> None:
    """Verify torch_like.fbgemm_pack_quantized_matrix records a symbolic Fbgemm_pack_quantized_matrix_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fbgemm_pack_quantized_matrix(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fbgemm_pack_quantized_matrix_out"
    assert out.dtype == DType.FLOAT32


def test_feature_alpha_dropout() -> None:
    """Verify torch_like.feature_alpha_dropout records a symbolic Feature_alpha_dropout_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.feature_alpha_dropout(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Feature_alpha_dropout_out"
    assert out.dtype == DType.FLOAT32


def test_feature_alpha_dropout_() -> None:
    """Verify torch_like.feature_alpha_dropout_ records a symbolic Feature_alpha_dropout__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.feature_alpha_dropout_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Feature_alpha_dropout__out"
    assert out.dtype == DType.FLOAT32


def test_feature_dropout() -> None:
    """Verify torch_like.feature_dropout records a symbolic Feature_dropout_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.feature_dropout(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Feature_dropout_out"
    assert out.dtype == DType.FLOAT32


def test_feature_dropout_() -> None:
    """Verify torch_like.feature_dropout_ records a symbolic Feature_dropout__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.feature_dropout_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Feature_dropout__out"
    assert out.dtype == DType.FLOAT32


def test_fused_moving_avg_obs_fake_quant() -> None:
    """Verify torch_like.fused_moving_avg_obs_fake_quant records a symbolic Fused_moving_avg_obs_fake_quant_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fused_moving_avg_obs_fake_quant(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fused_moving_avg_obs_fake_quant_out"
    assert out.dtype == DType.FLOAT32


def test_grid_sampler() -> None:
    """Verify torch_like.grid_sampler records a symbolic Grid_sampler_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.grid_sampler(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Grid_sampler_out"
    assert out.dtype == DType.FLOAT32


def test_grid_sampler_2d() -> None:
    """Verify torch_like.grid_sampler_2d records a symbolic Grid_sampler_2d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.grid_sampler_2d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Grid_sampler_2d_out"
    assert out.dtype == DType.FLOAT32


def test_grid_sampler_3d() -> None:
    """Verify torch_like.grid_sampler_3d records a symbolic Grid_sampler_3d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.grid_sampler_3d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Grid_sampler_3d_out"
    assert out.dtype == DType.FLOAT32


def test_group_norm() -> None:
    """Verify torch_like.group_norm records a symbolic Group_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.group_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Group_norm_out"
    assert out.dtype == DType.FLOAT32


def test_gru_cell() -> None:
    """Verify torch_like.gru_cell records a symbolic Gru_cell_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.gru_cell(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Gru_cell_out"
    assert out.dtype == DType.FLOAT32


def test_hardshrink() -> None:
    """Verify torch_like.hardshrink records a symbolic Hardshrink_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.hardshrink(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Hardshrink_out"
    assert out.dtype == DType.FLOAT32


def test_hinge_embedding_loss() -> None:
    """Verify torch_like.hinge_embedding_loss records a symbolic Hinge_embedding_loss_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.hinge_embedding_loss(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Hinge_embedding_loss_out"
    assert out.dtype == DType.FLOAT32


def test_instance_norm() -> None:
    """Verify torch_like.instance_norm records a symbolic Instance_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.instance_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Instance_norm_out"
    assert out.dtype == DType.FLOAT32


def test_int_repr() -> None:
    """Verify torch_like.int_repr records a symbolic Int_repr_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.int_repr(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Int_repr_out"
    assert out.dtype == DType.FLOAT32


def test_kl_div() -> None:
    """Verify torch_like.kl_div records a symbolic Kl_div_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.kl_div(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Kl_div_out"
    assert out.dtype == DType.FLOAT32


def test_layer_norm() -> None:
    """Verify torch_like.layer_norm records a symbolic Layer_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.layer_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Layer_norm_out"
    assert out.dtype == DType.FLOAT32


def test_lstm_cell() -> None:
    """Verify torch_like.lstm_cell records a symbolic Lstm_cell_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.lstm_cell(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Lstm_cell_out"
    assert out.dtype == DType.FLOAT32


def test_margin_ranking_loss() -> None:
    """Verify torch_like.margin_ranking_loss records a symbolic Margin_ranking_loss_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.margin_ranking_loss(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Margin_ranking_loss_out"
    assert out.dtype == DType.FLOAT32


def test_max_pool1d() -> None:
    """Verify torch_like.max_pool1d records a symbolic Max_pool1d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.max_pool1d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Max_pool1d_out"
    assert out.dtype == DType.FLOAT32


def test_max_pool1d_with_indices() -> None:
    """Verify torch_like.max_pool1d_with_indices records a symbolic Max_pool1d_with_indices_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.max_pool1d_with_indices(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Max_pool1d_with_indices_out"
    assert out.dtype == DType.FLOAT32


def test_max_pool2d() -> None:
    """Verify torch_like.max_pool2d records a symbolic Max_pool2d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.max_pool2d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Max_pool2d_out"
    assert out.dtype == DType.FLOAT32


def test_max_pool3d() -> None:
    """Verify torch_like.max_pool3d records a symbolic Max_pool3d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.max_pool3d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Max_pool3d_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_batch_norm() -> None:
    """Verify torch_like.miopen_batch_norm records a symbolic Miopen_batch_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.miopen_batch_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Miopen_batch_norm_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_convolution() -> None:
    """Verify torch_like.miopen_convolution records a symbolic Miopen_convolution_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.miopen_convolution(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Miopen_convolution_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_convolution_add_relu() -> None:
    """Verify torch_like.miopen_convolution_add_relu records a symbolic Miopen_convolution_add_relu_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.miopen_convolution_add_relu(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Miopen_convolution_add_relu_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_convolution_relu() -> None:
    """Verify torch_like.miopen_convolution_relu records a symbolic Miopen_convolution_relu_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.miopen_convolution_relu(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Miopen_convolution_relu_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_convolution_transpose() -> None:
    """Verify torch_like.miopen_convolution_transpose records a symbolic Miopen_convolution_transpose_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.miopen_convolution_transpose(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Miopen_convolution_transpose_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_ctc_loss() -> None:
    """Verify torch_like.miopen_ctc_loss records a symbolic Miopen_ctc_loss_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.miopen_ctc_loss(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Miopen_ctc_loss_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_depthwise_convolution() -> None:
    """Verify torch_like.miopen_depthwise_convolution records a symbolic Miopen_depthwise_convolution_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.miopen_depthwise_convolution(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Miopen_depthwise_convolution_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_rnn() -> None:
    """Verify torch_like.miopen_rnn records a symbolic Miopen_rnn_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.miopen_rnn(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Miopen_rnn_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_adaptive_avg_pool2d() -> None:
    """Verify torch_like.mkldnn_adaptive_avg_pool2d records a symbolic Mkldnn_adaptive_avg_pool2d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mkldnn_adaptive_avg_pool2d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mkldnn_adaptive_avg_pool2d_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_convolution() -> None:
    """Verify torch_like.mkldnn_convolution records a symbolic Mkldnn_convolution_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mkldnn_convolution(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mkldnn_convolution_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_linear_backward_weights() -> None:
    """Verify torch_like.mkldnn_linear_backward_weights records a symbolic Mkldnn_linear_backward_weights_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mkldnn_linear_backward_weights(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mkldnn_linear_backward_weights_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_max_pool2d() -> None:
    """Verify torch_like.mkldnn_max_pool2d records a symbolic Mkldnn_max_pool2d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mkldnn_max_pool2d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mkldnn_max_pool2d_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_max_pool3d() -> None:
    """Verify torch_like.mkldnn_max_pool3d records a symbolic Mkldnn_max_pool3d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mkldnn_max_pool3d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mkldnn_max_pool3d_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_rnn_layer() -> None:
    """Verify torch_like.mkldnn_rnn_layer records a symbolic Mkldnn_rnn_layer_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mkldnn_rnn_layer(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mkldnn_rnn_layer_out"
    assert out.dtype == DType.FLOAT32


def test_native_batch_norm() -> None:
    """Verify torch_like.native_batch_norm records a symbolic Native_batch_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.native_batch_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Native_batch_norm_out"
    assert out.dtype == DType.FLOAT32


def test_native_channel_shuffle() -> None:
    """Verify torch_like.native_channel_shuffle records a symbolic Native_channel_shuffle_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.native_channel_shuffle(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Native_channel_shuffle_out"
    assert out.dtype == DType.FLOAT32


def test_native_dropout() -> None:
    """Verify torch_like.native_dropout records a symbolic Native_dropout_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.native_dropout(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Native_dropout_out"
    assert out.dtype == DType.FLOAT32


def test_native_group_norm() -> None:
    """Verify torch_like.native_group_norm records a symbolic Native_group_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.native_group_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Native_group_norm_out"
    assert out.dtype == DType.FLOAT32


def test_native_layer_norm() -> None:
    """Verify torch_like.native_layer_norm records a symbolic Native_layer_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.native_layer_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Native_layer_norm_out"
    assert out.dtype == DType.FLOAT32


def test_native_norm() -> None:
    """Verify torch_like.native_norm records a symbolic Native_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.native_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Native_norm_out"
    assert out.dtype == DType.FLOAT32


def test_pairwise_distance() -> None:
    """Verify torch_like.pairwise_distance records a symbolic Pairwise_distance_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.pairwise_distance(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Pairwise_distance_out"
    assert out.dtype == DType.FLOAT32


def test_pdist() -> None:
    """Verify torch_like.pdist records a symbolic Pdist_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.pdist(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Pdist_out"
    assert out.dtype == DType.FLOAT32


def test_pixel_shuffle() -> None:
    """Verify torch_like.pixel_shuffle records a symbolic Pixel_shuffle_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.pixel_shuffle(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Pixel_shuffle_out"
    assert out.dtype == DType.FLOAT32


def test_pixel_unshuffle() -> None:
    """Verify torch_like.pixel_unshuffle records a symbolic Pixel_unshuffle_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.pixel_unshuffle(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Pixel_unshuffle_out"
    assert out.dtype == DType.FLOAT32


def test_poisson_nll_loss() -> None:
    """Verify torch_like.poisson_nll_loss records a symbolic Poisson_nll_loss_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.poisson_nll_loss(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Poisson_nll_loss_out"
    assert out.dtype == DType.FLOAT32


def test_q_per_channel_axis() -> None:
    """Verify torch_like.q_per_channel_axis records a symbolic Q_per_channel_axis_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.q_per_channel_axis(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Q_per_channel_axis_out"
    assert out.dtype == DType.FLOAT32


def test_q_per_channel_scales() -> None:
    """Verify torch_like.q_per_channel_scales records a symbolic Q_per_channel_scales_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.q_per_channel_scales(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Q_per_channel_scales_out"
    assert out.dtype == DType.FLOAT32


def test_q_per_channel_zero_points() -> None:
    """Verify torch_like.q_per_channel_zero_points records a symbolic Q_per_channel_zero_points_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.q_per_channel_zero_points(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Q_per_channel_zero_points_out"
    assert out.dtype == DType.FLOAT32


def test_q_scale() -> None:
    """Verify torch_like.q_scale records a symbolic Q_scale_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.q_scale(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Q_scale_out"
    assert out.dtype == DType.FLOAT32


def test_q_zero_point() -> None:
    """Verify torch_like.q_zero_point records a symbolic Q_zero_point_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.q_zero_point(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Q_zero_point_out"
    assert out.dtype == DType.FLOAT32


def test_quantize_per_channel() -> None:
    """Verify torch_like.quantize_per_channel records a symbolic Quantize_per_channel_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantize_per_channel(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantize_per_channel_out"
    assert out.dtype == DType.FLOAT32


def test_quantize_per_tensor() -> None:
    """Verify torch_like.quantize_per_tensor records a symbolic Quantize_per_tensor_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantize_per_tensor(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantize_per_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_quantize_per_tensor_dynamic() -> None:
    """Verify torch_like.quantize_per_tensor_dynamic records a symbolic Quantize_per_tensor_dynamic_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantize_per_tensor_dynamic(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantize_per_tensor_dynamic_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_batch_norm() -> None:
    """Verify torch_like.quantized_batch_norm records a symbolic Quantized_batch_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantized_batch_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantized_batch_norm_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_gru_cell() -> None:
    """Verify torch_like.quantized_gru_cell records a symbolic Quantized_gru_cell_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantized_gru_cell(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantized_gru_cell_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_lstm_cell() -> None:
    """Verify torch_like.quantized_lstm_cell records a symbolic Quantized_lstm_cell_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantized_lstm_cell(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantized_lstm_cell_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_max_pool1d() -> None:
    """Verify torch_like.quantized_max_pool1d records a symbolic Quantized_max_pool1d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantized_max_pool1d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantized_max_pool1d_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_max_pool2d() -> None:
    """Verify torch_like.quantized_max_pool2d records a symbolic Quantized_max_pool2d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantized_max_pool2d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantized_max_pool2d_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_max_pool3d() -> None:
    """Verify torch_like.quantized_max_pool3d records a symbolic Quantized_max_pool3d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantized_max_pool3d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantized_max_pool3d_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_rnn_relu_cell() -> None:
    """Verify torch_like.quantized_rnn_relu_cell records a symbolic Quantized_rnn_relu_cell_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantized_rnn_relu_cell(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantized_rnn_relu_cell_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_rnn_tanh_cell() -> None:
    """Verify torch_like.quantized_rnn_tanh_cell records a symbolic Quantized_rnn_tanh_cell_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantized_rnn_tanh_cell(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantized_rnn_tanh_cell_out"
    assert out.dtype == DType.FLOAT32


def test_relu_() -> None:
    """Verify torch_like.relu_ records a symbolic Relu__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.relu_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Relu__out"
    assert out.dtype == DType.FLOAT32


def test_rms_norm() -> None:
    """Verify torch_like.rms_norm records a symbolic Rms_norm_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rms_norm(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rms_norm_out"
    assert out.dtype == DType.FLOAT32


def test_rnn_relu() -> None:
    """Verify torch_like.rnn_relu records a symbolic Rnn_relu_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rnn_relu(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rnn_relu_out"
    assert out.dtype == DType.FLOAT32


def test_rnn_relu_cell() -> None:
    """Verify torch_like.rnn_relu_cell records a symbolic Rnn_relu_cell_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rnn_relu_cell(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rnn_relu_cell_out"
    assert out.dtype == DType.FLOAT32


def test_rnn_tanh() -> None:
    """Verify torch_like.rnn_tanh records a symbolic Rnn_tanh_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rnn_tanh(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rnn_tanh_out"
    assert out.dtype == DType.FLOAT32


def test_rnn_tanh_cell() -> None:
    """Verify torch_like.rnn_tanh_cell records a symbolic Rnn_tanh_cell_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rnn_tanh_cell(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rnn_tanh_cell_out"
    assert out.dtype == DType.FLOAT32


def test_rrelu() -> None:
    """Verify torch_like.rrelu records a symbolic Rrelu_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rrelu(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rrelu_out"
    assert out.dtype == DType.FLOAT32


def test_rrelu_() -> None:
    """Verify torch_like.rrelu_ records a symbolic Rrelu__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rrelu_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rrelu__out"
    assert out.dtype == DType.FLOAT32


def test_selu_() -> None:
    """Verify torch_like.selu_ records a symbolic Selu__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.selu_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Selu__out"
    assert out.dtype == DType.FLOAT32


def test_sigmoid_() -> None:
    """Verify torch_like.sigmoid_ records a symbolic Sigmoid__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sigmoid_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sigmoid__out"
    assert out.dtype == DType.FLOAT32


def test_tanh_() -> None:
    """Verify torch_like.tanh_ records a symbolic Tanh__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.tanh_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Tanh__out"
    assert out.dtype == DType.FLOAT32


def test_threshold() -> None:
    """Verify torch_like.threshold records a symbolic Threshold_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.threshold(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Threshold_out"
    assert out.dtype == DType.FLOAT32


def test_threshold_() -> None:
    """Verify torch_like.threshold_ records a symbolic Threshold__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.threshold_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Threshold__out"
    assert out.dtype == DType.FLOAT32


def test_triplet_margin_loss() -> None:
    """Verify torch_like.triplet_margin_loss records a symbolic Triplet_margin_loss_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.triplet_margin_loss(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Triplet_margin_loss_out"
    assert out.dtype == DType.FLOAT32


def test_all() -> None:
    """Verify torch_like.all records a symbolic All_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.all(x)
    assert isinstance(out, IRTensor)
    assert out.name == "All_out"
    assert out.dtype == DType.FLOAT32


def test_allclose() -> None:
    """Verify torch_like.allclose records a symbolic Allclose_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.allclose(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Allclose_out"
    assert out.dtype == DType.FLOAT32


def test_amax() -> None:
    """Verify torch_like.amax records a symbolic Amax_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.amax(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Amax_out"
    assert out.dtype == DType.FLOAT32


def test_amin() -> None:
    """Verify torch_like.amin records a symbolic Amin_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.amin(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Amin_out"
    assert out.dtype == DType.FLOAT32


def test_aminmax() -> None:
    """Verify torch_like.aminmax records a symbolic Aminmax_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.aminmax(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Aminmax_out"
    assert out.dtype == DType.FLOAT32


def test_any() -> None:
    """Verify torch_like.any records a symbolic Any_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.any(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Any_out"
    assert out.dtype == DType.FLOAT32


def test_bincount() -> None:
    """Verify torch_like.bincount records a symbolic Bincount_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bincount(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bincount_out"
    assert out.dtype == DType.FLOAT32


def test_count_nonzero() -> None:
    """Verify torch_like.count_nonzero records a symbolic Count_nonzero_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.count_nonzero(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Count_nonzero_out"
    assert out.dtype == DType.FLOAT32


def test_cummax() -> None:
    """Verify torch_like.cummax records a symbolic Cummax_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cummax(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cummax_out"
    assert out.dtype == DType.FLOAT32


def test_cummin() -> None:
    """Verify torch_like.cummin records a symbolic Cummin_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cummin(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cummin_out"
    assert out.dtype == DType.FLOAT32


def test_cumprod() -> None:
    """Verify torch_like.cumprod records a symbolic Cumprod_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cumprod(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cumprod_out"
    assert out.dtype == DType.FLOAT32


def test_cumulative_trapezoid() -> None:
    """Verify torch_like.cumulative_trapezoid records a symbolic Cumulative_trapezoid_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cumulative_trapezoid(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cumulative_trapezoid_out"
    assert out.dtype == DType.FLOAT32


def test_dist() -> None:
    """Verify torch_like.dist records a symbolic Dist_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.dist(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Dist_out"
    assert out.dtype == DType.FLOAT32


def test_histc() -> None:
    """Verify torch_like.histc records a symbolic Histc_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.histc(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Histc_out"
    assert out.dtype == DType.FLOAT32


def test_histogram() -> None:
    """Verify torch_like.histogram records a symbolic Histogram_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.histogram(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Histogram_out"
    assert out.dtype == DType.FLOAT32


def test_histogramdd() -> None:
    """Verify torch_like.histogramdd records a symbolic Histogramdd_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.histogramdd(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Histogramdd_out"
    assert out.dtype == DType.FLOAT32


def test_kthvalue() -> None:
    """Verify torch_like.kthvalue records a symbolic Kthvalue_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.kthvalue(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Kthvalue_out"
    assert out.dtype == DType.FLOAT32


def test_logcumsumexp() -> None:
    """Verify torch_like.logcumsumexp records a symbolic Logcumsumexp_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.logcumsumexp(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Logcumsumexp_out"
    assert out.dtype == DType.FLOAT32


def test_logsumexp() -> None:
    """Verify torch_like.logsumexp records a symbolic Logsumexp_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.logsumexp(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Logsumexp_out"
    assert out.dtype == DType.FLOAT32


def test_median() -> None:
    """Verify torch_like.median records a symbolic Median_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.median(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Median_out"
    assert out.dtype == DType.FLOAT32


def test_mode() -> None:
    """Verify torch_like.mode records a symbolic Mode_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.mode(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Mode_out"
    assert out.dtype == DType.FLOAT32


def test_nanmean() -> None:
    """Verify torch_like.nanmean records a symbolic Nanmean_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.nanmean(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Nanmean_out"
    assert out.dtype == DType.FLOAT32


def test_nanmedian() -> None:
    """Verify torch_like.nanmedian records a symbolic Nanmedian_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.nanmedian(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Nanmedian_out"
    assert out.dtype == DType.FLOAT32


def test_nanquantile() -> None:
    """Verify torch_like.nanquantile records a symbolic Nanquantile_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.nanquantile(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Nanquantile_out"
    assert out.dtype == DType.FLOAT32


def test_nansum() -> None:
    """Verify torch_like.nansum records a symbolic Nansum_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.nansum(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Nansum_out"
    assert out.dtype == DType.FLOAT32


def test_norm_except_dim() -> None:
    """Verify torch_like.norm_except_dim records a symbolic Norm_except_dim_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.norm_except_dim(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Norm_except_dim_out"
    assert out.dtype == DType.FLOAT32


def test_numel() -> None:
    """Verify torch_like.numel records a symbolic Numel_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.numel(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Numel_out"
    assert out.dtype == DType.FLOAT32


def test_prod() -> None:
    """Verify torch_like.prod records a symbolic Prod_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.prod(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Prod_out"
    assert out.dtype == DType.FLOAT32


def test_quantile() -> None:
    """Verify torch_like.quantile records a symbolic Quantile_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.quantile(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Quantile_out"
    assert out.dtype == DType.FLOAT32


def test_std() -> None:
    """Verify torch_like.std records a symbolic Std_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.std(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Std_out"
    assert out.dtype == DType.FLOAT32


def test_std_mean() -> None:
    """Verify torch_like.std_mean records a symbolic Std_mean_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.std_mean(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Std_mean_out"
    assert out.dtype == DType.FLOAT32


def test_trapezoid() -> None:
    """Verify torch_like.trapezoid records a symbolic Trapezoid_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.trapezoid(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Trapezoid_out"
    assert out.dtype == DType.FLOAT32


def test_trapz() -> None:
    """Verify torch_like.trapz records a symbolic Trapz_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.trapz(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Trapz_out"
    assert out.dtype == DType.FLOAT32


def test_unique_consecutive() -> None:
    """Verify torch_like.unique_consecutive records a symbolic Unique_consecutive_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unique_consecutive(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unique_consecutive_out"
    assert out.dtype == DType.FLOAT32


def test_var() -> None:
    """Verify torch_like.var records a symbolic Var_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.var(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Var_out"
    assert out.dtype == DType.FLOAT32


def test_var_mean() -> None:
    """Verify torch_like.var_mean records a symbolic Var_mean_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.var_mean(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Var_mean_out"
    assert out.dtype == DType.FLOAT32


def test_chunk() -> None:
    """Verify torch_like.chunk records a symbolic Chunk_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.chunk(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Chunk_out"
    assert out.dtype == DType.FLOAT32


def test_split() -> None:
    """Verify torch_like.split records a symbolic Split_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.split(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Split_out"
    assert out.dtype == DType.FLOAT32


def test_stack() -> None:
    """Verify torch_like.stack records a symbolic Stack_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.stack(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Stack_out"
    assert out.dtype == DType.FLOAT32


def test_unravel_index() -> None:
    """Verify torch_like.unravel_index records a symbolic Unravel_index_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unravel_index(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unravel_index_out"
    assert out.dtype == DType.FLOAT32


def test_alias_copy() -> None:
    """Verify torch_like.alias_copy records a symbolic Alias_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.alias_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Alias_copy_out"
    assert out.dtype == DType.FLOAT32


def test_align_tensors() -> None:
    """Verify torch_like.align_tensors records a symbolic Align_tensors_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.align_tensors(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Align_tensors_out"
    assert out.dtype == DType.FLOAT32


def test_argsort() -> None:
    """Verify torch_like.argsort records a symbolic Argsort_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.argsort(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Argsort_out"
    assert out.dtype == DType.FLOAT32


def test_argwhere() -> None:
    """Verify torch_like.argwhere records a symbolic Argwhere_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.argwhere(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Argwhere_out"
    assert out.dtype == DType.FLOAT32


def test_as_strided() -> None:
    """Verify torch_like.as_strided records a symbolic As_strided_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.as_strided(x)
    assert isinstance(out, IRTensor)
    assert out.name == "As_strided_out"
    assert out.dtype == DType.FLOAT32


def test_as_strided_() -> None:
    """Verify torch_like.as_strided_ records a symbolic As_strided__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.as_strided_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "As_strided__out"
    assert out.dtype == DType.FLOAT32


def test_as_strided_copy() -> None:
    """Verify torch_like.as_strided_copy records a symbolic As_strided_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.as_strided_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "As_strided_copy_out"
    assert out.dtype == DType.FLOAT32


def test_as_strided_scatter() -> None:
    """Verify torch_like.as_strided_scatter records a symbolic As_strided_scatter_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.as_strided_scatter(x)
    assert isinstance(out, IRTensor)
    assert out.name == "As_strided_scatter_out"
    assert out.dtype == DType.FLOAT32


def test_atleast_1d() -> None:
    """Verify torch_like.atleast_1d records a symbolic Atleast_1d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.atleast_1d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Atleast_1d_out"
    assert out.dtype == DType.FLOAT32


def test_atleast_2d() -> None:
    """Verify torch_like.atleast_2d records a symbolic Atleast_2d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.atleast_2d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Atleast_2d_out"
    assert out.dtype == DType.FLOAT32


def test_atleast_3d() -> None:
    """Verify torch_like.atleast_3d records a symbolic Atleast_3d_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.atleast_3d(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Atleast_3d_out"
    assert out.dtype == DType.FLOAT32


def test_block_diag() -> None:
    """Verify torch_like.block_diag records a symbolic Block_diag_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.block_diag(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Block_diag_out"
    assert out.dtype == DType.FLOAT32


def test_broadcast_tensors() -> None:
    """Verify torch_like.broadcast_tensors records a symbolic Broadcast_tensors_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.broadcast_tensors(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Broadcast_tensors_out"
    assert out.dtype == DType.FLOAT32


def test_broadcast_to() -> None:
    """Verify torch_like.broadcast_to records a symbolic Broadcast_to_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.broadcast_to(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Broadcast_to_out"
    assert out.dtype == DType.FLOAT32


def test_bucketize() -> None:
    """Verify torch_like.bucketize records a symbolic Bucketize_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.bucketize(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Bucketize_out"
    assert out.dtype == DType.FLOAT32


def test_cartesian_prod() -> None:
    """Verify torch_like.cartesian_prod records a symbolic Cartesian_prod_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cartesian_prod(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cartesian_prod_out"
    assert out.dtype == DType.FLOAT32


def test_cat() -> None:
    """Verify torch_like.cat records a symbolic Cat_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.cat(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Cat_out"
    assert out.dtype == DType.FLOAT32


def test_ccol_indices_copy() -> None:
    """Verify torch_like.ccol_indices_copy records a symbolic Ccol_indices_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ccol_indices_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ccol_indices_copy_out"
    assert out.dtype == DType.FLOAT32


def test_clone() -> None:
    """Verify torch_like.clone records a symbolic Clone_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.clone(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Clone_out"
    assert out.dtype == DType.FLOAT32


def test_col_indices_copy() -> None:
    """Verify torch_like.col_indices_copy records a symbolic Col_indices_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.col_indices_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Col_indices_copy_out"
    assert out.dtype == DType.FLOAT32


def test_column_stack() -> None:
    """Verify torch_like.column_stack records a symbolic Column_stack_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.column_stack(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Column_stack_out"
    assert out.dtype == DType.FLOAT32


def test_combinations() -> None:
    """Verify torch_like.combinations records a symbolic Combinations_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.combinations(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Combinations_out"
    assert out.dtype == DType.FLOAT32


def test_concatenate() -> None:
    """Verify torch_like.concatenate records a symbolic Concatenate_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.concatenate(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Concatenate_out"
    assert out.dtype == DType.FLOAT32


def test_crow_indices_copy() -> None:
    """Verify torch_like.crow_indices_copy records a symbolic Crow_indices_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.crow_indices_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Crow_indices_copy_out"
    assert out.dtype == DType.FLOAT32


def test_detach() -> None:
    """Verify torch_like.detach records a symbolic Detach_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.detach(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Detach_out"
    assert out.dtype == DType.FLOAT32


def test_detach_() -> None:
    """Verify torch_like.detach_ records a symbolic Detach__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.detach_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Detach__out"
    assert out.dtype == DType.FLOAT32


def test_detach_copy() -> None:
    """Verify torch_like.detach_copy records a symbolic Detach_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.detach_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Detach_copy_out"
    assert out.dtype == DType.FLOAT32


def test_dsplit() -> None:
    """Verify torch_like.dsplit records a symbolic Dsplit_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.dsplit(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Dsplit_out"
    assert out.dtype == DType.FLOAT32


def test_dstack() -> None:
    """Verify torch_like.dstack records a symbolic Dstack_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.dstack(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Dstack_out"
    assert out.dtype == DType.FLOAT32


def test_expand_copy() -> None:
    """Verify torch_like.expand_copy records a symbolic Expand_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.expand_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Expand_copy_out"
    assert out.dtype == DType.FLOAT32


def test_flatten() -> None:
    """Verify torch_like.flatten records a symbolic Flatten_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.flatten(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Flatten_out"
    assert out.dtype == DType.FLOAT32


def test_flip() -> None:
    """Verify torch_like.flip records a symbolic Flip_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.flip(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Flip_out"
    assert out.dtype == DType.FLOAT32


def test_fliplr() -> None:
    """Verify torch_like.fliplr records a symbolic Fliplr_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.fliplr(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Fliplr_out"
    assert out.dtype == DType.FLOAT32


def test_flipud() -> None:
    """Verify torch_like.flipud records a symbolic Flipud_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.flipud(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Flipud_out"
    assert out.dtype == DType.FLOAT32


def test_hsplit() -> None:
    """Verify torch_like.hsplit records a symbolic Hsplit_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.hsplit(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Hsplit_out"
    assert out.dtype == DType.FLOAT32


def test_hstack() -> None:
    """Verify torch_like.hstack records a symbolic Hstack_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.hstack(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Hstack_out"
    assert out.dtype == DType.FLOAT32


def test_index_add() -> None:
    """Verify torch_like.index_add records a symbolic Index_add_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.index_add(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Index_add_out"
    assert out.dtype == DType.FLOAT32


def test_index_copy() -> None:
    """Verify torch_like.index_copy records a symbolic Index_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.index_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Index_copy_out"
    assert out.dtype == DType.FLOAT32


def test_index_fill() -> None:
    """Verify torch_like.index_fill records a symbolic Index_fill_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.index_fill(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Index_fill_out"
    assert out.dtype == DType.FLOAT32


def test_index_put() -> None:
    """Verify torch_like.index_put records a symbolic Index_put_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.index_put(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Index_put_out"
    assert out.dtype == DType.FLOAT32


def test_index_put_() -> None:
    """Verify torch_like.index_put_ records a symbolic Index_put__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.index_put_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Index_put__out"
    assert out.dtype == DType.FLOAT32


def test_index_reduce() -> None:
    """Verify torch_like.index_reduce records a symbolic Index_reduce_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.index_reduce(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Index_reduce_out"
    assert out.dtype == DType.FLOAT32


def test_index_select() -> None:
    """Verify torch_like.index_select records a symbolic Index_select_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.index_select(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Index_select_out"
    assert out.dtype == DType.FLOAT32


def test_indices_copy() -> None:
    """Verify torch_like.indices_copy records a symbolic Indices_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.indices_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Indices_copy_out"
    assert out.dtype == DType.FLOAT32


def test_isin() -> None:
    """Verify torch_like.isin records a symbolic Isin_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.isin(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Isin_out"
    assert out.dtype == DType.FLOAT32


def test_masked_fill() -> None:
    """Verify torch_like.masked_fill records a symbolic Masked_fill_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.masked_fill(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Masked_fill_out"
    assert out.dtype == DType.FLOAT32


def test_masked_scatter() -> None:
    """Verify torch_like.masked_scatter records a symbolic Masked_scatter_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.masked_scatter(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Masked_scatter_out"
    assert out.dtype == DType.FLOAT32


def test_masked_select() -> None:
    """Verify torch_like.masked_select records a symbolic Masked_select_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.masked_select(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Masked_select_out"
    assert out.dtype == DType.FLOAT32


def test_meshgrid() -> None:
    """Verify torch_like.meshgrid records a symbolic Meshgrid_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.meshgrid(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Meshgrid_out"
    assert out.dtype == DType.FLOAT32


def test_moveaxis() -> None:
    """Verify torch_like.moveaxis records a symbolic Moveaxis_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.moveaxis(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Moveaxis_out"
    assert out.dtype == DType.FLOAT32


def test_movedim() -> None:
    """Verify torch_like.movedim records a symbolic Movedim_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.movedim(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Movedim_out"
    assert out.dtype == DType.FLOAT32


def test_msort() -> None:
    """Verify torch_like.msort records a symbolic Msort_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.msort(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Msort_out"
    assert out.dtype == DType.FLOAT32


def test_narrow() -> None:
    """Verify torch_like.narrow records a symbolic Narrow_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.narrow(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Narrow_out"
    assert out.dtype == DType.FLOAT32


def test_narrow_copy() -> None:
    """Verify torch_like.narrow_copy records a symbolic Narrow_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.narrow_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Narrow_copy_out"
    assert out.dtype == DType.FLOAT32


def test_nonzero() -> None:
    """Verify torch_like.nonzero records a symbolic Nonzero_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.nonzero(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Nonzero_out"
    assert out.dtype == DType.FLOAT32


def test_nonzero_static() -> None:
    """Verify torch_like.nonzero_static records a symbolic Nonzero_static_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.nonzero_static(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Nonzero_static_out"
    assert out.dtype == DType.FLOAT32


def test_permute() -> None:
    """Verify torch_like.permute records a symbolic Permute_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.permute(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Permute_out"
    assert out.dtype == DType.FLOAT32


def test_permute_copy() -> None:
    """Verify torch_like.permute_copy records a symbolic Permute_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.permute_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Permute_copy_out"
    assert out.dtype == DType.FLOAT32


def test_put() -> None:
    """Verify torch_like.put records a symbolic Put_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.put(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Put_out"
    assert out.dtype == DType.FLOAT32


def test_ravel() -> None:
    """Verify torch_like.ravel records a symbolic Ravel_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.ravel(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Ravel_out"
    assert out.dtype == DType.FLOAT32


def test_repeat_interleave() -> None:
    """Verify torch_like.repeat_interleave records a symbolic Repeat_interleave_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.repeat_interleave(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Repeat_interleave_out"
    assert out.dtype == DType.FLOAT32


def test_resize_as_() -> None:
    """Verify torch_like.resize_as_ records a symbolic Resize_as__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.resize_as_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Resize_as__out"
    assert out.dtype == DType.FLOAT32


def test_resize_as_sparse_() -> None:
    """Verify torch_like.resize_as_sparse_ records a symbolic Resize_as_sparse__out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.resize_as_sparse_(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Resize_as_sparse__out"
    assert out.dtype == DType.FLOAT32


def test_roll() -> None:
    """Verify torch_like.roll records a symbolic Roll_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.roll(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Roll_out"
    assert out.dtype == DType.FLOAT32


def test_rot90() -> None:
    """Verify torch_like.rot90 records a symbolic Rot90_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.rot90(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Rot90_out"
    assert out.dtype == DType.FLOAT32


def test_row_indices_copy() -> None:
    """Verify torch_like.row_indices_copy records a symbolic Row_indices_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.row_indices_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Row_indices_copy_out"
    assert out.dtype == DType.FLOAT32


def test_row_stack() -> None:
    """Verify torch_like.row_stack records a symbolic Row_stack_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.row_stack(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Row_stack_out"
    assert out.dtype == DType.FLOAT32


def test_scatter_add() -> None:
    """Verify torch_like.scatter_add records a symbolic Scatter_add_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.scatter_add(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Scatter_add_out"
    assert out.dtype == DType.FLOAT32


def test_scatter_reduce() -> None:
    """Verify torch_like.scatter_reduce records a symbolic Scatter_reduce_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.scatter_reduce(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Scatter_reduce_out"
    assert out.dtype == DType.FLOAT32


def test_searchsorted() -> None:
    """Verify torch_like.searchsorted records a symbolic Searchsorted_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.searchsorted(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Searchsorted_out"
    assert out.dtype == DType.FLOAT32


def test_select() -> None:
    """Verify torch_like.select records a symbolic Select_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.select(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Select_out"
    assert out.dtype == DType.FLOAT32


def test_select_copy() -> None:
    """Verify torch_like.select_copy records a symbolic Select_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.select_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Select_copy_out"
    assert out.dtype == DType.FLOAT32


def test_select_scatter() -> None:
    """Verify torch_like.select_scatter records a symbolic Select_scatter_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.select_scatter(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Select_scatter_out"
    assert out.dtype == DType.FLOAT32


def test_slice_copy() -> None:
    """Verify torch_like.slice_copy records a symbolic Slice_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.slice_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Slice_copy_out"
    assert out.dtype == DType.FLOAT32


def test_slice_inverse() -> None:
    """Verify torch_like.slice_inverse records a symbolic Slice_inverse_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.slice_inverse(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Slice_inverse_out"
    assert out.dtype == DType.FLOAT32


def test_slice_scatter() -> None:
    """Verify torch_like.slice_scatter records a symbolic Slice_scatter_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.slice_scatter(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Slice_scatter_out"
    assert out.dtype == DType.FLOAT32


def test_sort() -> None:
    """Verify torch_like.sort records a symbolic Sort_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.sort(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Sort_out"
    assert out.dtype == DType.FLOAT32


def test_split_copy() -> None:
    """Verify torch_like.split_copy records a symbolic Split_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.split_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Split_copy_out"
    assert out.dtype == DType.FLOAT32


def test_split_with_sizes() -> None:
    """Verify torch_like.split_with_sizes records a symbolic Split_with_sizes_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.split_with_sizes(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Split_with_sizes_out"
    assert out.dtype == DType.FLOAT32


def test_split_with_sizes_copy() -> None:
    """Verify torch_like.split_with_sizes_copy records a symbolic Split_with_sizes_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.split_with_sizes_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Split_with_sizes_copy_out"
    assert out.dtype == DType.FLOAT32


def test_squeeze() -> None:
    """Verify torch_like.squeeze records a symbolic Squeeze_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.squeeze(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Squeeze_out"
    assert out.dtype == DType.FLOAT32


def test_squeeze_copy() -> None:
    """Verify torch_like.squeeze_copy records a symbolic Squeeze_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.squeeze_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Squeeze_copy_out"
    assert out.dtype == DType.FLOAT32


def test_swapaxes() -> None:
    """Verify torch_like.swapaxes records a symbolic Swapaxes_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.swapaxes(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Swapaxes_out"
    assert out.dtype == DType.FLOAT32


def test_swapdims() -> None:
    """Verify torch_like.swapdims records a symbolic Swapdims_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.swapdims(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Swapdims_out"
    assert out.dtype == DType.FLOAT32


def test_t() -> None:
    """Verify torch_like.t records a symbolic T_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.t(x)
    assert isinstance(out, IRTensor)
    assert out.name == "T_out"
    assert out.dtype == DType.FLOAT32


def test_t_copy() -> None:
    """Verify torch_like.t_copy records a symbolic T_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.t_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "T_copy_out"
    assert out.dtype == DType.FLOAT32


def test_take() -> None:
    """Verify torch_like.take records a symbolic Take_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.take(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Take_out"
    assert out.dtype == DType.FLOAT32


def test_take_along_dim() -> None:
    """Verify torch_like.take_along_dim records a symbolic Take_along_dim_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.take_along_dim(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Take_along_dim_out"
    assert out.dtype == DType.FLOAT32


def test_tensor_split() -> None:
    """Verify torch_like.tensor_split records a symbolic Tensor_split_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.tensor_split(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Tensor_split_out"
    assert out.dtype == DType.FLOAT32


def test_transpose_copy() -> None:
    """Verify torch_like.transpose_copy records a symbolic Transpose_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.transpose_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Transpose_copy_out"
    assert out.dtype == DType.FLOAT32


def test_unbind() -> None:
    """Verify torch_like.unbind records a symbolic Unbind_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unbind(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unbind_out"
    assert out.dtype == DType.FLOAT32


def test_unbind_copy() -> None:
    """Verify torch_like.unbind_copy records a symbolic Unbind_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unbind_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unbind_copy_out"
    assert out.dtype == DType.FLOAT32


def test_unflatten() -> None:
    """Verify torch_like.unflatten records a symbolic Unflatten_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unflatten(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unflatten_out"
    assert out.dtype == DType.FLOAT32


def test_unfold_copy() -> None:
    """Verify torch_like.unfold_copy records a symbolic Unfold_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unfold_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unfold_copy_out"
    assert out.dtype == DType.FLOAT32


def test_unsafe_chunk() -> None:
    """Verify torch_like.unsafe_chunk records a symbolic Unsafe_chunk_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unsafe_chunk(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unsafe_chunk_out"
    assert out.dtype == DType.FLOAT32


def test_unsafe_split() -> None:
    """Verify torch_like.unsafe_split records a symbolic Unsafe_split_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unsafe_split(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unsafe_split_out"
    assert out.dtype == DType.FLOAT32


def test_unsafe_split_with_sizes() -> None:
    """Verify torch_like.unsafe_split_with_sizes records a symbolic Unsafe_split_with_sizes_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unsafe_split_with_sizes(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unsafe_split_with_sizes_out"
    assert out.dtype == DType.FLOAT32


def test_unsqueeze() -> None:
    """Verify torch_like.unsqueeze records a symbolic Unsqueeze_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unsqueeze(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unsqueeze_out"
    assert out.dtype == DType.FLOAT32


def test_unsqueeze_copy() -> None:
    """Verify torch_like.unsqueeze_copy records a symbolic Unsqueeze_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.unsqueeze_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Unsqueeze_copy_out"
    assert out.dtype == DType.FLOAT32


def test_values_copy() -> None:
    """Verify torch_like.values_copy records a symbolic Values_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.values_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Values_copy_out"
    assert out.dtype == DType.FLOAT32


def test_view_as_complex() -> None:
    """Verify torch_like.view_as_complex records a symbolic View_as_complex_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.view_as_complex(x)
    assert isinstance(out, IRTensor)
    assert out.name == "View_as_complex_out"
    assert out.dtype == DType.FLOAT32


def test_view_as_complex_copy() -> None:
    """Verify torch_like.view_as_complex_copy records a symbolic View_as_complex_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.view_as_complex_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "View_as_complex_copy_out"
    assert out.dtype == DType.FLOAT32


def test_view_as_real() -> None:
    """Verify torch_like.view_as_real records a symbolic View_as_real_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.view_as_real(x)
    assert isinstance(out, IRTensor)
    assert out.name == "View_as_real_out"
    assert out.dtype == DType.FLOAT32


def test_view_as_real_copy() -> None:
    """Verify torch_like.view_as_real_copy records a symbolic View_as_real_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.view_as_real_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "View_as_real_copy_out"
    assert out.dtype == DType.FLOAT32


def test_view_copy() -> None:
    """Verify torch_like.view_copy records a symbolic View_copy_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.view_copy(x)
    assert isinstance(out, IRTensor)
    assert out.name == "View_copy_out"
    assert out.dtype == DType.FLOAT32


def test_vsplit() -> None:
    """Verify torch_like.vsplit records a symbolic Vsplit_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.vsplit(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Vsplit_out"
    assert out.dtype == DType.FLOAT32


def test_vstack() -> None:
    """Verify torch_like.vstack records a symbolic Vstack_out tensor."""
    x = IRTensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_like.vstack(x)
    assert isinstance(out, IRTensor)
    assert out.name == "Vstack_out"
    assert out.dtype == DType.FLOAT32
