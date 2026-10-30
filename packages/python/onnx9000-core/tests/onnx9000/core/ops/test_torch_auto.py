"""Tests for PyTorch ATen symbolic operator helpers in torch_auto."""

from __future__ import annotations

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Tensor
from onnx9000.core.ops import torch_auto


def test_chunk() -> None:
    """Verify chunk records a symbolic Chunk tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.chunk(x)
    assert isinstance(out, Tensor)
    assert out.name == "Chunk_out"
    assert out.dtype == DType.FLOAT32


def test_lobpcg() -> None:
    """Verify lobpcg records a symbolic Lobpcg tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.lobpcg(x)
    assert isinstance(out, Tensor)
    assert out.name == "Lobpcg_out"
    assert out.dtype == DType.FLOAT32


def test_rand() -> None:
    """Verify rand records a symbolic Rand tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rand(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rand_out"
    assert out.dtype == DType.FLOAT32


def test_split() -> None:
    """Verify split records a symbolic Split tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.split(x)
    assert isinstance(out, Tensor)
    assert out.name == "Split_out"
    assert out.dtype == DType.FLOAT32


def test_stack() -> None:
    """Verify stack records a symbolic Stack tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.stack(x)
    assert isinstance(out, Tensor)
    assert out.name == "Stack_out"
    assert out.dtype == DType.FLOAT32


def test_unravel_index() -> None:
    """Verify unravel_index records a symbolic Unravel_index tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unravel_index(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unravel_index_out"
    assert out.dtype == DType.FLOAT32


def test_abs_() -> None:
    """Verify abs_ records a symbolic Abs_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.abs_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Abs__out"
    assert out.dtype == DType.FLOAT32


def test_absolute() -> None:
    """Verify absolute records a symbolic Absolute tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.absolute(x)
    assert isinstance(out, Tensor)
    assert out.name == "Absolute_out"
    assert out.dtype == DType.FLOAT32


def test_acos_() -> None:
    """Verify acos_ records a symbolic Acos_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.acos_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Acos__out"
    assert out.dtype == DType.FLOAT32


def test_acosh_() -> None:
    """Verify acosh_ records a symbolic Acosh_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.acosh_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Acosh__out"
    assert out.dtype == DType.FLOAT32


def test_adaptive_avg_pool1d() -> None:
    """Verify adaptive_avg_pool1d records a symbolic Adaptive_avg_pool1d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.adaptive_avg_pool1d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Adaptive_avg_pool1d_out"
    assert out.dtype == DType.FLOAT32


def test_adaptive_max_pool1d() -> None:
    """Verify adaptive_max_pool1d records a symbolic Adaptive_max_pool1d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.adaptive_max_pool1d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Adaptive_max_pool1d_out"
    assert out.dtype == DType.FLOAT32


def test_addbmm() -> None:
    """Verify addbmm records a symbolic Addbmm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.addbmm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Addbmm_out"
    assert out.dtype == DType.FLOAT32


def test_addcdiv() -> None:
    """Verify addcdiv records a symbolic Addcdiv tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.addcdiv(x)
    assert isinstance(out, Tensor)
    assert out.name == "Addcdiv_out"
    assert out.dtype == DType.FLOAT32


def test_addcmul() -> None:
    """Verify addcmul records a symbolic Addcmul tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.addcmul(x)
    assert isinstance(out, Tensor)
    assert out.name == "Addcmul_out"
    assert out.dtype == DType.FLOAT32


def test_addmm() -> None:
    """Verify addmm records a symbolic Addmm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.addmm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Addmm_out"
    assert out.dtype == DType.FLOAT32


def test_addmv() -> None:
    """Verify addmv records a symbolic Addmv tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.addmv(x)
    assert isinstance(out, Tensor)
    assert out.name == "Addmv_out"
    assert out.dtype == DType.FLOAT32


def test_addmv_() -> None:
    """Verify addmv_ records a symbolic Addmv_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.addmv_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Addmv__out"
    assert out.dtype == DType.FLOAT32


def test_addr() -> None:
    """Verify addr records a symbolic Addr tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.addr(x)
    assert isinstance(out, Tensor)
    assert out.name == "Addr_out"
    assert out.dtype == DType.FLOAT32


def test_adjoint() -> None:
    """Verify adjoint records a symbolic Adjoint tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.adjoint(x)
    assert isinstance(out, Tensor)
    assert out.name == "Adjoint_out"
    assert out.dtype == DType.FLOAT32


def test_affine_grid_generator() -> None:
    """Verify affine_grid_generator records a symbolic Affine_grid_generator tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.affine_grid_generator(x)
    assert isinstance(out, Tensor)
    assert out.name == "Affine_grid_generator_out"
    assert out.dtype == DType.FLOAT32


def test_alias_copy() -> None:
    """Verify alias_copy records a symbolic Alias_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.alias_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Alias_copy_out"
    assert out.dtype == DType.FLOAT32


def test_align_tensors() -> None:
    """Verify align_tensors records a symbolic Align_tensors tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.align_tensors(x)
    assert isinstance(out, Tensor)
    assert out.name == "Align_tensors_out"
    assert out.dtype == DType.FLOAT32


def test_all() -> None:
    """Verify all records a symbolic All tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.all(x)
    assert isinstance(out, Tensor)
    assert out.name == "All_out"
    assert out.dtype == DType.FLOAT32


def test_allclose() -> None:
    """Verify allclose records a symbolic Allclose tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.allclose(x)
    assert isinstance(out, Tensor)
    assert out.name == "Allclose_out"
    assert out.dtype == DType.FLOAT32


def test_alpha_dropout() -> None:
    """Verify alpha_dropout records a symbolic Alpha_dropout tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.alpha_dropout(x)
    assert isinstance(out, Tensor)
    assert out.name == "Alpha_dropout_out"
    assert out.dtype == DType.FLOAT32


def test_alpha_dropout_() -> None:
    """Verify alpha_dropout_ records a symbolic Alpha_dropout_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.alpha_dropout_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Alpha_dropout__out"
    assert out.dtype == DType.FLOAT32


def test_amax() -> None:
    """Verify amax records a symbolic Amax tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.amax(x)
    assert isinstance(out, Tensor)
    assert out.name == "Amax_out"
    assert out.dtype == DType.FLOAT32


def test_amin() -> None:
    """Verify amin records a symbolic Amin tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.amin(x)
    assert isinstance(out, Tensor)
    assert out.name == "Amin_out"
    assert out.dtype == DType.FLOAT32


def test_aminmax() -> None:
    """Verify aminmax records a symbolic Aminmax tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.aminmax(x)
    assert isinstance(out, Tensor)
    assert out.name == "Aminmax_out"
    assert out.dtype == DType.FLOAT32


def test_angle() -> None:
    """Verify angle records a symbolic Angle tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.angle(x)
    assert isinstance(out, Tensor)
    assert out.name == "Angle_out"
    assert out.dtype == DType.FLOAT32


def test_any() -> None:
    """Verify any records a symbolic Any tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.any(x)
    assert isinstance(out, Tensor)
    assert out.name == "Any_out"
    assert out.dtype == DType.FLOAT32


def test_arange() -> None:
    """Verify arange records a symbolic Arange tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arange(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arange_out"
    assert out.dtype == DType.FLOAT32


def test_arccos() -> None:
    """Verify arccos records a symbolic Arccos tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arccos(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arccos_out"
    assert out.dtype == DType.FLOAT32


def test_arccos_() -> None:
    """Verify arccos_ records a symbolic Arccos_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arccos_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arccos__out"
    assert out.dtype == DType.FLOAT32


def test_arccosh() -> None:
    """Verify arccosh records a symbolic Arccosh tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arccosh(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arccosh_out"
    assert out.dtype == DType.FLOAT32


def test_arccosh_() -> None:
    """Verify arccosh_ records a symbolic Arccosh_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arccosh_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arccosh__out"
    assert out.dtype == DType.FLOAT32


def test_arcsin() -> None:
    """Verify arcsin records a symbolic Arcsin tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arcsin(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arcsin_out"
    assert out.dtype == DType.FLOAT32


def test_arcsin_() -> None:
    """Verify arcsin_ records a symbolic Arcsin_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arcsin_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arcsin__out"
    assert out.dtype == DType.FLOAT32


def test_arcsinh() -> None:
    """Verify arcsinh records a symbolic Arcsinh tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arcsinh(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arcsinh_out"
    assert out.dtype == DType.FLOAT32


def test_arcsinh_() -> None:
    """Verify arcsinh_ records a symbolic Arcsinh_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arcsinh_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arcsinh__out"
    assert out.dtype == DType.FLOAT32


def test_arctan() -> None:
    """Verify arctan records a symbolic Arctan tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arctan(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arctan_out"
    assert out.dtype == DType.FLOAT32


def test_arctan2() -> None:
    """Verify arctan2 records a symbolic Arctan2 tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arctan2(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arctan2_out"
    assert out.dtype == DType.FLOAT32


def test_arctan_() -> None:
    """Verify arctan_ records a symbolic Arctan_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arctan_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arctan__out"
    assert out.dtype == DType.FLOAT32


def test_arctanh() -> None:
    """Verify arctanh records a symbolic Arctanh tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arctanh(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arctanh_out"
    assert out.dtype == DType.FLOAT32


def test_arctanh_() -> None:
    """Verify arctanh_ records a symbolic Arctanh_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.arctanh_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Arctanh__out"
    assert out.dtype == DType.FLOAT32


def test_argsort() -> None:
    """Verify argsort records a symbolic Argsort tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.argsort(x)
    assert isinstance(out, Tensor)
    assert out.name == "Argsort_out"
    assert out.dtype == DType.FLOAT32


def test_argwhere() -> None:
    """Verify argwhere records a symbolic Argwhere tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.argwhere(x)
    assert isinstance(out, Tensor)
    assert out.name == "Argwhere_out"
    assert out.dtype == DType.FLOAT32


def test_as_strided() -> None:
    """Verify as_strided records a symbolic As_strided tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.as_strided(x)
    assert isinstance(out, Tensor)
    assert out.name == "As_strided_out"
    assert out.dtype == DType.FLOAT32


def test_as_strided_() -> None:
    """Verify as_strided_ records a symbolic As_strided_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.as_strided_(x)
    assert isinstance(out, Tensor)
    assert out.name == "As_strided__out"
    assert out.dtype == DType.FLOAT32


def test_as_strided_copy() -> None:
    """Verify as_strided_copy records a symbolic As_strided_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.as_strided_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "As_strided_copy_out"
    assert out.dtype == DType.FLOAT32


def test_as_strided_scatter() -> None:
    """Verify as_strided_scatter records a symbolic As_strided_scatter tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.as_strided_scatter(x)
    assert isinstance(out, Tensor)
    assert out.name == "As_strided_scatter_out"
    assert out.dtype == DType.FLOAT32


def test_as_tensor() -> None:
    """Verify as_tensor records a symbolic As_tensor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.as_tensor(x)
    assert isinstance(out, Tensor)
    assert out.name == "As_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_asarray() -> None:
    """Verify asarray records a symbolic Asarray tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.asarray(x)
    assert isinstance(out, Tensor)
    assert out.name == "Asarray_out"
    assert out.dtype == DType.FLOAT32


def test_asin_() -> None:
    """Verify asin_ records a symbolic Asin_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.asin_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Asin__out"
    assert out.dtype == DType.FLOAT32


def test_asinh_() -> None:
    """Verify asinh_ records a symbolic Asinh_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.asinh_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Asinh__out"
    assert out.dtype == DType.FLOAT32


def test_atan2() -> None:
    """Verify atan2 records a symbolic Atan2 tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.atan2(x)
    assert isinstance(out, Tensor)
    assert out.name == "Atan2_out"
    assert out.dtype == DType.FLOAT32


def test_atan_() -> None:
    """Verify atan_ records a symbolic Atan_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.atan_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Atan__out"
    assert out.dtype == DType.FLOAT32


def test_atanh_() -> None:
    """Verify atanh_ records a symbolic Atanh_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.atanh_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Atanh__out"
    assert out.dtype == DType.FLOAT32


def test_atleast_1d() -> None:
    """Verify atleast_1d records a symbolic Atleast_1d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.atleast_1d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Atleast_1d_out"
    assert out.dtype == DType.FLOAT32


def test_atleast_2d() -> None:
    """Verify atleast_2d records a symbolic Atleast_2d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.atleast_2d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Atleast_2d_out"
    assert out.dtype == DType.FLOAT32


def test_atleast_3d() -> None:
    """Verify atleast_3d records a symbolic Atleast_3d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.atleast_3d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Atleast_3d_out"
    assert out.dtype == DType.FLOAT32


def test_avg_pool1d() -> None:
    """Verify avg_pool1d records a symbolic Avg_pool1d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.avg_pool1d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Avg_pool1d_out"
    assert out.dtype == DType.FLOAT32


def test_baddbmm() -> None:
    """Verify baddbmm records a symbolic Baddbmm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.baddbmm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Baddbmm_out"
    assert out.dtype == DType.FLOAT32


def test_bartlett_window() -> None:
    """Verify bartlett_window records a symbolic Bartlett_window tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.bartlett_window(x)
    assert isinstance(out, Tensor)
    assert out.name == "Bartlett_window_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm() -> None:
    """Verify batch_norm records a symbolic Batch_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.batch_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Batch_norm_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_backward_elemt() -> None:
    """Verify batch_norm_backward_elemt records a symbolic Batch_norm_backward_elemt tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.batch_norm_backward_elemt(x)
    assert isinstance(out, Tensor)
    assert out.name == "Batch_norm_backward_elemt_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_backward_reduce() -> None:
    """Verify batch_norm_backward_reduce records a symbolic Batch_norm_backward_reduce tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.batch_norm_backward_reduce(x)
    assert isinstance(out, Tensor)
    assert out.name == "Batch_norm_backward_reduce_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_elemt() -> None:
    """Verify batch_norm_elemt records a symbolic Batch_norm_elemt tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.batch_norm_elemt(x)
    assert isinstance(out, Tensor)
    assert out.name == "Batch_norm_elemt_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_gather_stats() -> None:
    """Verify batch_norm_gather_stats records a symbolic Batch_norm_gather_stats tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.batch_norm_gather_stats(x)
    assert isinstance(out, Tensor)
    assert out.name == "Batch_norm_gather_stats_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_gather_stats_with_counts() -> None:
    """Verify batch_norm_gather_stats_with_counts records a symbolic Batch_norm_gather_stats_with_counts tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.batch_norm_gather_stats_with_counts(x)
    assert isinstance(out, Tensor)
    assert out.name == "Batch_norm_gather_stats_with_counts_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_stats() -> None:
    """Verify batch_norm_stats records a symbolic Batch_norm_stats tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.batch_norm_stats(x)
    assert isinstance(out, Tensor)
    assert out.name == "Batch_norm_stats_out"
    assert out.dtype == DType.FLOAT32


def test_batch_norm_update_stats() -> None:
    """Verify batch_norm_update_stats records a symbolic Batch_norm_update_stats tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.batch_norm_update_stats(x)
    assert isinstance(out, Tensor)
    assert out.name == "Batch_norm_update_stats_out"
    assert out.dtype == DType.FLOAT32


def test_bilinear() -> None:
    """Verify bilinear records a symbolic Bilinear tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.bilinear(x)
    assert isinstance(out, Tensor)
    assert out.name == "Bilinear_out"
    assert out.dtype == DType.FLOAT32


def test_binary_cross_entropy_with_logits() -> None:
    """Verify binary_cross_entropy_with_logits records a symbolic Binary_cross_entropy_with_logits tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.binary_cross_entropy_with_logits(x)
    assert isinstance(out, Tensor)
    assert out.name == "Binary_cross_entropy_with_logits_out"
    assert out.dtype == DType.FLOAT32


def test_bincount() -> None:
    """Verify bincount records a symbolic Bincount tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.bincount(x)
    assert isinstance(out, Tensor)
    assert out.name == "Bincount_out"
    assert out.dtype == DType.FLOAT32


def test_binomial() -> None:
    """Verify binomial records a symbolic Binomial tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.binomial(x)
    assert isinstance(out, Tensor)
    assert out.name == "Binomial_out"
    assert out.dtype == DType.FLOAT32


def test_bitwise_left_shift() -> None:
    """Verify bitwise_left_shift records a symbolic Bitwise_left_shift tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.bitwise_left_shift(x)
    assert isinstance(out, Tensor)
    assert out.name == "Bitwise_left_shift_out"
    assert out.dtype == DType.FLOAT32


def test_bitwise_right_shift() -> None:
    """Verify bitwise_right_shift records a symbolic Bitwise_right_shift tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.bitwise_right_shift(x)
    assert isinstance(out, Tensor)
    assert out.name == "Bitwise_right_shift_out"
    assert out.dtype == DType.FLOAT32


def test_block_diag() -> None:
    """Verify block_diag records a symbolic Block_diag tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.block_diag(x)
    assert isinstance(out, Tensor)
    assert out.name == "Block_diag_out"
    assert out.dtype == DType.FLOAT32


def test_bmm() -> None:
    """Verify bmm records a symbolic Bmm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.bmm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Bmm_out"
    assert out.dtype == DType.FLOAT32


def test_broadcast_tensors() -> None:
    """Verify broadcast_tensors records a symbolic Broadcast_tensors tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.broadcast_tensors(x)
    assert isinstance(out, Tensor)
    assert out.name == "Broadcast_tensors_out"
    assert out.dtype == DType.FLOAT32


def test_broadcast_to() -> None:
    """Verify broadcast_to records a symbolic Broadcast_to tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.broadcast_to(x)
    assert isinstance(out, Tensor)
    assert out.name == "Broadcast_to_out"
    assert out.dtype == DType.FLOAT32


def test_bucketize() -> None:
    """Verify bucketize records a symbolic Bucketize tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.bucketize(x)
    assert isinstance(out, Tensor)
    assert out.name == "Bucketize_out"
    assert out.dtype == DType.FLOAT32


def test_cartesian_prod() -> None:
    """Verify cartesian_prod records a symbolic Cartesian_prod tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cartesian_prod(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cartesian_prod_out"
    assert out.dtype == DType.FLOAT32


def test_cat() -> None:
    """Verify cat records a symbolic Cat tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cat(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cat_out"
    assert out.dtype == DType.FLOAT32


def test_ccol_indices_copy() -> None:
    """Verify ccol_indices_copy records a symbolic Ccol_indices_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ccol_indices_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ccol_indices_copy_out"
    assert out.dtype == DType.FLOAT32


def test_cdist() -> None:
    """Verify cdist records a symbolic Cdist tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cdist(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cdist_out"
    assert out.dtype == DType.FLOAT32


def test_ceil_() -> None:
    """Verify ceil_ records a symbolic Ceil_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ceil_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ceil__out"
    assert out.dtype == DType.FLOAT32


def test_celu_() -> None:
    """Verify celu_ records a symbolic Celu_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.celu_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Celu__out"
    assert out.dtype == DType.FLOAT32


def test_chain_matmul() -> None:
    """Verify chain_matmul records a symbolic Chain_matmul tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.chain_matmul(x)
    assert isinstance(out, Tensor)
    assert out.name == "Chain_matmul_out"
    assert out.dtype == DType.FLOAT32


def test_channel_shuffle() -> None:
    """Verify channel_shuffle records a symbolic Channel_shuffle tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.channel_shuffle(x)
    assert isinstance(out, Tensor)
    assert out.name == "Channel_shuffle_out"
    assert out.dtype == DType.FLOAT32


def test_cholesky() -> None:
    """Verify cholesky records a symbolic Cholesky tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cholesky(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cholesky_out"
    assert out.dtype == DType.FLOAT32


def test_cholesky_inverse() -> None:
    """Verify cholesky_inverse records a symbolic Cholesky_inverse tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cholesky_inverse(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cholesky_inverse_out"
    assert out.dtype == DType.FLOAT32


def test_cholesky_solve() -> None:
    """Verify cholesky_solve records a symbolic Cholesky_solve tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cholesky_solve(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cholesky_solve_out"
    assert out.dtype == DType.FLOAT32


def test_choose_qparams_optimized() -> None:
    """Verify choose_qparams_optimized records a symbolic Choose_qparams_optimized tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.choose_qparams_optimized(x)
    assert isinstance(out, Tensor)
    assert out.name == "Choose_qparams_optimized_out"
    assert out.dtype == DType.FLOAT32


def test_clamp() -> None:
    """Verify clamp records a symbolic Clamp tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.clamp(x)
    assert isinstance(out, Tensor)
    assert out.name == "Clamp_out"
    assert out.dtype == DType.FLOAT32


def test_clamp_() -> None:
    """Verify clamp_ records a symbolic Clamp_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.clamp_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Clamp__out"
    assert out.dtype == DType.FLOAT32


def test_clamp_max() -> None:
    """Verify clamp_max records a symbolic Clamp_max tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.clamp_max(x)
    assert isinstance(out, Tensor)
    assert out.name == "Clamp_max_out"
    assert out.dtype == DType.FLOAT32


def test_clamp_max_() -> None:
    """Verify clamp_max_ records a symbolic Clamp_max_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.clamp_max_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Clamp_max__out"
    assert out.dtype == DType.FLOAT32


def test_clamp_min() -> None:
    """Verify clamp_min records a symbolic Clamp_min tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.clamp_min(x)
    assert isinstance(out, Tensor)
    assert out.name == "Clamp_min_out"
    assert out.dtype == DType.FLOAT32


def test_clamp_min_() -> None:
    """Verify clamp_min_ records a symbolic Clamp_min_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.clamp_min_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Clamp_min__out"
    assert out.dtype == DType.FLOAT32


def test_clip_() -> None:
    """Verify clip_ records a symbolic Clip_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.clip_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Clip__out"
    assert out.dtype == DType.FLOAT32


def test_clone() -> None:
    """Verify clone records a symbolic Clone tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.clone(x)
    assert isinstance(out, Tensor)
    assert out.name == "Clone_out"
    assert out.dtype == DType.FLOAT32


def test_col_indices_copy() -> None:
    """Verify col_indices_copy records a symbolic Col_indices_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.col_indices_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Col_indices_copy_out"
    assert out.dtype == DType.FLOAT32


def test_column_stack() -> None:
    """Verify column_stack records a symbolic Column_stack tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.column_stack(x)
    assert isinstance(out, Tensor)
    assert out.name == "Column_stack_out"
    assert out.dtype == DType.FLOAT32


def test_combinations() -> None:
    """Verify combinations records a symbolic Combinations tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.combinations(x)
    assert isinstance(out, Tensor)
    assert out.name == "Combinations_out"
    assert out.dtype == DType.FLOAT32


def test_complex() -> None:
    """Verify complex records a symbolic Complex tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.complex(x)
    assert isinstance(out, Tensor)
    assert out.name == "Complex_out"
    assert out.dtype == DType.FLOAT32


def test_concatenate() -> None:
    """Verify concatenate records a symbolic Concatenate tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.concatenate(x)
    assert isinstance(out, Tensor)
    assert out.name == "Concatenate_out"
    assert out.dtype == DType.FLOAT32


def test_conj() -> None:
    """Verify conj records a symbolic Conj tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.conj(x)
    assert isinstance(out, Tensor)
    assert out.name == "Conj_out"
    assert out.dtype == DType.FLOAT32


def test_conj_physical() -> None:
    """Verify conj_physical records a symbolic Conj_physical tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.conj_physical(x)
    assert isinstance(out, Tensor)
    assert out.name == "Conj_physical_out"
    assert out.dtype == DType.FLOAT32


def test_conj_physical_() -> None:
    """Verify conj_physical_ records a symbolic Conj_physical_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.conj_physical_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Conj_physical__out"
    assert out.dtype == DType.FLOAT32


def test_constant_pad_nd() -> None:
    """Verify constant_pad_nd records a symbolic Constant_pad_nd tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.constant_pad_nd(x)
    assert isinstance(out, Tensor)
    assert out.name == "Constant_pad_nd_out"
    assert out.dtype == DType.FLOAT32


def test_conv1d() -> None:
    """Verify conv1d records a symbolic Conv1d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.conv1d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Conv1d_out"
    assert out.dtype == DType.FLOAT32


def test_conv2d() -> None:
    """Verify conv2d records a symbolic Conv2d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.conv2d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Conv2d_out"
    assert out.dtype == DType.FLOAT32


def test_conv3d() -> None:
    """Verify conv3d records a symbolic Conv3d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.conv3d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Conv3d_out"
    assert out.dtype == DType.FLOAT32


def test_conv_tbc() -> None:
    """Verify conv_tbc records a symbolic Conv_tbc tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.conv_tbc(x)
    assert isinstance(out, Tensor)
    assert out.name == "Conv_tbc_out"
    assert out.dtype == DType.FLOAT32


def test_conv_transpose1d() -> None:
    """Verify conv_transpose1d records a symbolic Conv_transpose1d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.conv_transpose1d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Conv_transpose1d_out"
    assert out.dtype == DType.FLOAT32


def test_conv_transpose2d() -> None:
    """Verify conv_transpose2d records a symbolic Conv_transpose2d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.conv_transpose2d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Conv_transpose2d_out"
    assert out.dtype == DType.FLOAT32


def test_conv_transpose3d() -> None:
    """Verify conv_transpose3d records a symbolic Conv_transpose3d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.conv_transpose3d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Conv_transpose3d_out"
    assert out.dtype == DType.FLOAT32


def test_convolution() -> None:
    """Verify convolution records a symbolic Convolution tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.convolution(x)
    assert isinstance(out, Tensor)
    assert out.name == "Convolution_out"
    assert out.dtype == DType.FLOAT32


def test_copysign() -> None:
    """Verify copysign records a symbolic Copysign tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.copysign(x)
    assert isinstance(out, Tensor)
    assert out.name == "Copysign_out"
    assert out.dtype == DType.FLOAT32


def test_corrcoef() -> None:
    """Verify corrcoef records a symbolic Corrcoef tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.corrcoef(x)
    assert isinstance(out, Tensor)
    assert out.name == "Corrcoef_out"
    assert out.dtype == DType.FLOAT32


def test_cos_() -> None:
    """Verify cos_ records a symbolic Cos_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cos_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cos__out"
    assert out.dtype == DType.FLOAT32


def test_cosh_() -> None:
    """Verify cosh_ records a symbolic Cosh_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cosh_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cosh__out"
    assert out.dtype == DType.FLOAT32


def test_cosine_embedding_loss() -> None:
    """Verify cosine_embedding_loss records a symbolic Cosine_embedding_loss tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cosine_embedding_loss(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cosine_embedding_loss_out"
    assert out.dtype == DType.FLOAT32


def test_cosine_similarity() -> None:
    """Verify cosine_similarity records a symbolic Cosine_similarity tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cosine_similarity(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cosine_similarity_out"
    assert out.dtype == DType.FLOAT32


def test_count_nonzero() -> None:
    """Verify count_nonzero records a symbolic Count_nonzero tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.count_nonzero(x)
    assert isinstance(out, Tensor)
    assert out.name == "Count_nonzero_out"
    assert out.dtype == DType.FLOAT32


def test_cov() -> None:
    """Verify cov records a symbolic Cov tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cov(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cov_out"
    assert out.dtype == DType.FLOAT32


def test_cross() -> None:
    """Verify cross records a symbolic Cross tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cross(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cross_out"
    assert out.dtype == DType.FLOAT32


def test_crow_indices_copy() -> None:
    """Verify crow_indices_copy records a symbolic Crow_indices_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.crow_indices_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Crow_indices_copy_out"
    assert out.dtype == DType.FLOAT32


def test_ctc_loss() -> None:
    """Verify ctc_loss records a symbolic Ctc_loss tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ctc_loss(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ctc_loss_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_affine_grid_generator() -> None:
    """Verify cudnn_affine_grid_generator records a symbolic Cudnn_affine_grid_generator tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cudnn_affine_grid_generator(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cudnn_affine_grid_generator_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_batch_norm() -> None:
    """Verify cudnn_batch_norm records a symbolic Cudnn_batch_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cudnn_batch_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cudnn_batch_norm_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_convolution() -> None:
    """Verify cudnn_convolution records a symbolic Cudnn_convolution tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cudnn_convolution(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cudnn_convolution_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_convolution_add_relu() -> None:
    """Verify cudnn_convolution_add_relu records a symbolic Cudnn_convolution_add_relu tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cudnn_convolution_add_relu(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cudnn_convolution_add_relu_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_convolution_relu() -> None:
    """Verify cudnn_convolution_relu records a symbolic Cudnn_convolution_relu tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cudnn_convolution_relu(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cudnn_convolution_relu_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_convolution_transpose() -> None:
    """Verify cudnn_convolution_transpose records a symbolic Cudnn_convolution_transpose tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cudnn_convolution_transpose(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cudnn_convolution_transpose_out"
    assert out.dtype == DType.FLOAT32


def test_cudnn_grid_sampler() -> None:
    """Verify cudnn_grid_sampler records a symbolic Cudnn_grid_sampler tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cudnn_grid_sampler(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cudnn_grid_sampler_out"
    assert out.dtype == DType.FLOAT32


def test_cummax() -> None:
    """Verify cummax records a symbolic Cummax tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cummax(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cummax_out"
    assert out.dtype == DType.FLOAT32


def test_cummin() -> None:
    """Verify cummin records a symbolic Cummin tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cummin(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cummin_out"
    assert out.dtype == DType.FLOAT32


def test_cumprod() -> None:
    """Verify cumprod records a symbolic Cumprod tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cumprod(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cumprod_out"
    assert out.dtype == DType.FLOAT32


def test_cumulative_trapezoid() -> None:
    """Verify cumulative_trapezoid records a symbolic Cumulative_trapezoid tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.cumulative_trapezoid(x)
    assert isinstance(out, Tensor)
    assert out.name == "Cumulative_trapezoid_out"
    assert out.dtype == DType.FLOAT32


def test_deg2rad() -> None:
    """Verify deg2rad records a symbolic Deg2rad tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.deg2rad(x)
    assert isinstance(out, Tensor)
    assert out.name == "Deg2rad_out"
    assert out.dtype == DType.FLOAT32


def test_deg2rad_() -> None:
    """Verify deg2rad_ records a symbolic Deg2rad_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.deg2rad_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Deg2rad__out"
    assert out.dtype == DType.FLOAT32


def test_dequantize() -> None:
    """Verify dequantize records a symbolic Dequantize tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.dequantize(x)
    assert isinstance(out, Tensor)
    assert out.name == "Dequantize_out"
    assert out.dtype == DType.FLOAT32


def test_detach() -> None:
    """Verify detach records a symbolic Detach tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.detach(x)
    assert isinstance(out, Tensor)
    assert out.name == "Detach_out"
    assert out.dtype == DType.FLOAT32


def test_detach_() -> None:
    """Verify detach_ records a symbolic Detach_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.detach_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Detach__out"
    assert out.dtype == DType.FLOAT32


def test_detach_copy() -> None:
    """Verify detach_copy records a symbolic Detach_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.detach_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Detach_copy_out"
    assert out.dtype == DType.FLOAT32


def test_diag() -> None:
    """Verify diag records a symbolic Diag tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.diag(x)
    assert isinstance(out, Tensor)
    assert out.name == "Diag_out"
    assert out.dtype == DType.FLOAT32


def test_diag_embed() -> None:
    """Verify diag_embed records a symbolic Diag_embed tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.diag_embed(x)
    assert isinstance(out, Tensor)
    assert out.name == "Diag_embed_out"
    assert out.dtype == DType.FLOAT32


def test_diagflat() -> None:
    """Verify diagflat records a symbolic Diagflat tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.diagflat(x)
    assert isinstance(out, Tensor)
    assert out.name == "Diagflat_out"
    assert out.dtype == DType.FLOAT32


def test_diagonal() -> None:
    """Verify diagonal records a symbolic Diagonal tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.diagonal(x)
    assert isinstance(out, Tensor)
    assert out.name == "Diagonal_out"
    assert out.dtype == DType.FLOAT32


def test_diagonal_copy() -> None:
    """Verify diagonal_copy records a symbolic Diagonal_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.diagonal_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Diagonal_copy_out"
    assert out.dtype == DType.FLOAT32


def test_diagonal_scatter() -> None:
    """Verify diagonal_scatter records a symbolic Diagonal_scatter tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.diagonal_scatter(x)
    assert isinstance(out, Tensor)
    assert out.name == "Diagonal_scatter_out"
    assert out.dtype == DType.FLOAT32


def test_diff() -> None:
    """Verify diff records a symbolic Diff tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.diff(x)
    assert isinstance(out, Tensor)
    assert out.name == "Diff_out"
    assert out.dtype == DType.FLOAT32


def test_digamma() -> None:
    """Verify digamma records a symbolic Digamma tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.digamma(x)
    assert isinstance(out, Tensor)
    assert out.name == "Digamma_out"
    assert out.dtype == DType.FLOAT32


def test_dist() -> None:
    """Verify dist records a symbolic Dist tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.dist(x)
    assert isinstance(out, Tensor)
    assert out.name == "Dist_out"
    assert out.dtype == DType.FLOAT32


def test_divide() -> None:
    """Verify divide records a symbolic Divide tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.divide(x)
    assert isinstance(out, Tensor)
    assert out.name == "Divide_out"
    assert out.dtype == DType.FLOAT32


def test_dot() -> None:
    """Verify dot records a symbolic Dot tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.dot(x)
    assert isinstance(out, Tensor)
    assert out.name == "Dot_out"
    assert out.dtype == DType.FLOAT32


def test_dropout_() -> None:
    """Verify dropout_ records a symbolic Dropout_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.dropout_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Dropout__out"
    assert out.dtype == DType.FLOAT32


def test_dsmm() -> None:
    """Verify dsmm records a symbolic Dsmm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.dsmm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Dsmm_out"
    assert out.dtype == DType.FLOAT32


def test_dsplit() -> None:
    """Verify dsplit records a symbolic Dsplit tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.dsplit(x)
    assert isinstance(out, Tensor)
    assert out.name == "Dsplit_out"
    assert out.dtype == DType.FLOAT32


def test_dstack() -> None:
    """Verify dstack records a symbolic Dstack tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.dstack(x)
    assert isinstance(out, Tensor)
    assert out.name == "Dstack_out"
    assert out.dtype == DType.FLOAT32


def test_embedding() -> None:
    """Verify embedding records a symbolic Embedding tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.embedding(x)
    assert isinstance(out, Tensor)
    assert out.name == "Embedding_out"
    assert out.dtype == DType.FLOAT32


def test_embedding_bag() -> None:
    """Verify embedding_bag records a symbolic Embedding_bag tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.embedding_bag(x)
    assert isinstance(out, Tensor)
    assert out.name == "Embedding_bag_out"
    assert out.dtype == DType.FLOAT32


def test_embedding_renorm_() -> None:
    """Verify embedding_renorm_ records a symbolic Embedding_renorm_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.embedding_renorm_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Embedding_renorm__out"
    assert out.dtype == DType.FLOAT32


def test_empty() -> None:
    """Verify empty records a symbolic Empty tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.empty(x)
    assert isinstance(out, Tensor)
    assert out.name == "Empty_out"
    assert out.dtype == DType.FLOAT32


def test_empty_like() -> None:
    """Verify empty_like records a symbolic Empty_like tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.empty_like(x)
    assert isinstance(out, Tensor)
    assert out.name == "Empty_like_out"
    assert out.dtype == DType.FLOAT32


def test_empty_permuted() -> None:
    """Verify empty_permuted records a symbolic Empty_permuted tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.empty_permuted(x)
    assert isinstance(out, Tensor)
    assert out.name == "Empty_permuted_out"
    assert out.dtype == DType.FLOAT32


def test_empty_quantized() -> None:
    """Verify empty_quantized records a symbolic Empty_quantized tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.empty_quantized(x)
    assert isinstance(out, Tensor)
    assert out.name == "Empty_quantized_out"
    assert out.dtype == DType.FLOAT32


def test_empty_strided() -> None:
    """Verify empty_strided records a symbolic Empty_strided tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.empty_strided(x)
    assert isinstance(out, Tensor)
    assert out.name == "Empty_strided_out"
    assert out.dtype == DType.FLOAT32


def test_eq() -> None:
    """Verify eq records a symbolic Eq tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.eq(x)
    assert isinstance(out, Tensor)
    assert out.name == "Eq_out"
    assert out.dtype == DType.FLOAT32


def test_erf_() -> None:
    """Verify erf_ records a symbolic Erf_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.erf_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Erf__out"
    assert out.dtype == DType.FLOAT32


def test_erfc() -> None:
    """Verify erfc records a symbolic Erfc tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.erfc(x)
    assert isinstance(out, Tensor)
    assert out.name == "Erfc_out"
    assert out.dtype == DType.FLOAT32


def test_erfc_() -> None:
    """Verify erfc_ records a symbolic Erfc_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.erfc_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Erfc__out"
    assert out.dtype == DType.FLOAT32


def test_erfinv() -> None:
    """Verify erfinv records a symbolic Erfinv tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.erfinv(x)
    assert isinstance(out, Tensor)
    assert out.name == "Erfinv_out"
    assert out.dtype == DType.FLOAT32


def test_exp2() -> None:
    """Verify exp2 records a symbolic Exp2 tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.exp2(x)
    assert isinstance(out, Tensor)
    assert out.name == "Exp2_out"
    assert out.dtype == DType.FLOAT32


def test_exp2_() -> None:
    """Verify exp2_ records a symbolic Exp2_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.exp2_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Exp2__out"
    assert out.dtype == DType.FLOAT32


def test_exp_() -> None:
    """Verify exp_ records a symbolic Exp_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.exp_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Exp__out"
    assert out.dtype == DType.FLOAT32


def test_expand_copy() -> None:
    """Verify expand_copy records a symbolic Expand_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.expand_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Expand_copy_out"
    assert out.dtype == DType.FLOAT32


def test_expm1() -> None:
    """Verify expm1 records a symbolic Expm1 tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.expm1(x)
    assert isinstance(out, Tensor)
    assert out.name == "Expm1_out"
    assert out.dtype == DType.FLOAT32


def test_expm1_() -> None:
    """Verify expm1_ records a symbolic Expm1_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.expm1_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Expm1__out"
    assert out.dtype == DType.FLOAT32


def test_eye() -> None:
    """Verify eye records a symbolic Eye tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.eye(x)
    assert isinstance(out, Tensor)
    assert out.name == "Eye_out"
    assert out.dtype == DType.FLOAT32


def test_fake_quantize_per_channel_affine() -> None:
    """Verify fake_quantize_per_channel_affine records a symbolic Fake_quantize_per_channel_affine tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fake_quantize_per_channel_affine(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fake_quantize_per_channel_affine_out"
    assert out.dtype == DType.FLOAT32


def test_fake_quantize_per_tensor_affine() -> None:
    """Verify fake_quantize_per_tensor_affine records a symbolic Fake_quantize_per_tensor_affine tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fake_quantize_per_tensor_affine(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fake_quantize_per_tensor_affine_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_linear_fp16_weight() -> None:
    """Verify fbgemm_linear_fp16_weight records a symbolic Fbgemm_linear_fp16_weight tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fbgemm_linear_fp16_weight(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fbgemm_linear_fp16_weight_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_linear_fp16_weight_fp32_activation() -> None:
    """Verify fbgemm_linear_fp16_weight_fp32_activation records a symbolic Fbgemm_linear_fp16_weight_fp32_activation tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fbgemm_linear_fp16_weight_fp32_activation(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fbgemm_linear_fp16_weight_fp32_activation_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_linear_int8_weight() -> None:
    """Verify fbgemm_linear_int8_weight records a symbolic Fbgemm_linear_int8_weight tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fbgemm_linear_int8_weight(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fbgemm_linear_int8_weight_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_linear_int8_weight_fp32_activation() -> None:
    """Verify fbgemm_linear_int8_weight_fp32_activation records a symbolic Fbgemm_linear_int8_weight_fp32_activation tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fbgemm_linear_int8_weight_fp32_activation(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fbgemm_linear_int8_weight_fp32_activation_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_linear_quantize_weight() -> None:
    """Verify fbgemm_linear_quantize_weight records a symbolic Fbgemm_linear_quantize_weight tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fbgemm_linear_quantize_weight(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fbgemm_linear_quantize_weight_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_pack_gemm_matrix_fp16() -> None:
    """Verify fbgemm_pack_gemm_matrix_fp16 records a symbolic Fbgemm_pack_gemm_matrix_fp16 tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fbgemm_pack_gemm_matrix_fp16(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fbgemm_pack_gemm_matrix_fp16_out"
    assert out.dtype == DType.FLOAT32


def test_fbgemm_pack_quantized_matrix() -> None:
    """Verify fbgemm_pack_quantized_matrix records a symbolic Fbgemm_pack_quantized_matrix tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fbgemm_pack_quantized_matrix(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fbgemm_pack_quantized_matrix_out"
    assert out.dtype == DType.FLOAT32


def test_feature_alpha_dropout() -> None:
    """Verify feature_alpha_dropout records a symbolic Feature_alpha_dropout tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.feature_alpha_dropout(x)
    assert isinstance(out, Tensor)
    assert out.name == "Feature_alpha_dropout_out"
    assert out.dtype == DType.FLOAT32


def test_feature_alpha_dropout_() -> None:
    """Verify feature_alpha_dropout_ records a symbolic Feature_alpha_dropout_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.feature_alpha_dropout_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Feature_alpha_dropout__out"
    assert out.dtype == DType.FLOAT32


def test_feature_dropout() -> None:
    """Verify feature_dropout records a symbolic Feature_dropout tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.feature_dropout(x)
    assert isinstance(out, Tensor)
    assert out.name == "Feature_dropout_out"
    assert out.dtype == DType.FLOAT32


def test_feature_dropout_() -> None:
    """Verify feature_dropout_ records a symbolic Feature_dropout_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.feature_dropout_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Feature_dropout__out"
    assert out.dtype == DType.FLOAT32


def test_fill() -> None:
    """Verify fill records a symbolic Fill tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fill(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fill_out"
    assert out.dtype == DType.FLOAT32


def test_fill_() -> None:
    """Verify fill_ records a symbolic Fill_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fill_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fill__out"
    assert out.dtype == DType.FLOAT32


def test_fix() -> None:
    """Verify fix records a symbolic Fix tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fix(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fix_out"
    assert out.dtype == DType.FLOAT32


def test_fix_() -> None:
    """Verify fix_ records a symbolic Fix_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fix_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fix__out"
    assert out.dtype == DType.FLOAT32


def test_flatten() -> None:
    """Verify flatten records a symbolic Flatten tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.flatten(x)
    assert isinstance(out, Tensor)
    assert out.name == "Flatten_out"
    assert out.dtype == DType.FLOAT32


def test_flip() -> None:
    """Verify flip records a symbolic Flip tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.flip(x)
    assert isinstance(out, Tensor)
    assert out.name == "Flip_out"
    assert out.dtype == DType.FLOAT32


def test_fliplr() -> None:
    """Verify fliplr records a symbolic Fliplr tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fliplr(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fliplr_out"
    assert out.dtype == DType.FLOAT32


def test_flipud() -> None:
    """Verify flipud records a symbolic Flipud tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.flipud(x)
    assert isinstance(out, Tensor)
    assert out.name == "Flipud_out"
    assert out.dtype == DType.FLOAT32


def test_float_power() -> None:
    """Verify float_power records a symbolic Float_power tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.float_power(x)
    assert isinstance(out, Tensor)
    assert out.name == "Float_power_out"
    assert out.dtype == DType.FLOAT32


def test_floor_() -> None:
    """Verify floor_ records a symbolic Floor_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.floor_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Floor__out"
    assert out.dtype == DType.FLOAT32


def test_floor_divide() -> None:
    """Verify floor_divide records a symbolic Floor_divide tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.floor_divide(x)
    assert isinstance(out, Tensor)
    assert out.name == "Floor_divide_out"
    assert out.dtype == DType.FLOAT32


def test_fmax() -> None:
    """Verify fmax records a symbolic Fmax tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fmax(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fmax_out"
    assert out.dtype == DType.FLOAT32


def test_fmin() -> None:
    """Verify fmin records a symbolic Fmin tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fmin(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fmin_out"
    assert out.dtype == DType.FLOAT32


def test_fmod() -> None:
    """Verify fmod records a symbolic Fmod tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fmod(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fmod_out"
    assert out.dtype == DType.FLOAT32


def test_frac() -> None:
    """Verify frac records a symbolic Frac tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.frac(x)
    assert isinstance(out, Tensor)
    assert out.name == "Frac_out"
    assert out.dtype == DType.FLOAT32


def test_frac_() -> None:
    """Verify frac_ records a symbolic Frac_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.frac_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Frac__out"
    assert out.dtype == DType.FLOAT32


def test_frexp() -> None:
    """Verify frexp records a symbolic Frexp tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.frexp(x)
    assert isinstance(out, Tensor)
    assert out.name == "Frexp_out"
    assert out.dtype == DType.FLOAT32


def test_frobenius_norm() -> None:
    """Verify frobenius_norm records a symbolic Frobenius_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.frobenius_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Frobenius_norm_out"
    assert out.dtype == DType.FLOAT32


def test_from_file() -> None:
    """Verify from_file records a symbolic From_file tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.from_file(x)
    assert isinstance(out, Tensor)
    assert out.name == "From_file_out"
    assert out.dtype == DType.FLOAT32


def test_from_numpy() -> None:
    """Verify from_numpy records a symbolic From_numpy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.from_numpy(x)
    assert isinstance(out, Tensor)
    assert out.name == "From_numpy_out"
    assert out.dtype == DType.FLOAT32


def test_frombuffer() -> None:
    """Verify frombuffer records a symbolic Frombuffer tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.frombuffer(x)
    assert isinstance(out, Tensor)
    assert out.name == "Frombuffer_out"
    assert out.dtype == DType.FLOAT32


def test_full() -> None:
    """Verify full records a symbolic Full tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.full(x)
    assert isinstance(out, Tensor)
    assert out.name == "Full_out"
    assert out.dtype == DType.FLOAT32


def test_full_like() -> None:
    """Verify full_like records a symbolic Full_like tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.full_like(x)
    assert isinstance(out, Tensor)
    assert out.name == "Full_like_out"
    assert out.dtype == DType.FLOAT32


def test_fused_moving_avg_obs_fake_quant() -> None:
    """Verify fused_moving_avg_obs_fake_quant records a symbolic Fused_moving_avg_obs_fake_quant tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.fused_moving_avg_obs_fake_quant(x)
    assert isinstance(out, Tensor)
    assert out.name == "Fused_moving_avg_obs_fake_quant_out"
    assert out.dtype == DType.FLOAT32


def test_gcd() -> None:
    """Verify gcd records a symbolic Gcd tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.gcd(x)
    assert isinstance(out, Tensor)
    assert out.name == "Gcd_out"
    assert out.dtype == DType.FLOAT32


def test_gcd_() -> None:
    """Verify gcd_ records a symbolic Gcd_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.gcd_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Gcd__out"
    assert out.dtype == DType.FLOAT32


def test_ge() -> None:
    """Verify ge records a symbolic Ge tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ge(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ge_out"
    assert out.dtype == DType.FLOAT32


def test_geqrf() -> None:
    """Verify geqrf records a symbolic Geqrf tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.geqrf(x)
    assert isinstance(out, Tensor)
    assert out.name == "Geqrf_out"
    assert out.dtype == DType.FLOAT32


def test_ger() -> None:
    """Verify ger records a symbolic Ger tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ger(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ger_out"
    assert out.dtype == DType.FLOAT32


def test_gradient() -> None:
    """Verify gradient records a symbolic Gradient tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.gradient(x)
    assert isinstance(out, Tensor)
    assert out.name == "Gradient_out"
    assert out.dtype == DType.FLOAT32


def test_greater_equal() -> None:
    """Verify greater_equal records a symbolic Greater_equal tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.greater_equal(x)
    assert isinstance(out, Tensor)
    assert out.name == "Greater_equal_out"
    assert out.dtype == DType.FLOAT32


def test_grid_sampler() -> None:
    """Verify grid_sampler records a symbolic Grid_sampler tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.grid_sampler(x)
    assert isinstance(out, Tensor)
    assert out.name == "Grid_sampler_out"
    assert out.dtype == DType.FLOAT32


def test_grid_sampler_2d() -> None:
    """Verify grid_sampler_2d records a symbolic Grid_sampler_2d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.grid_sampler_2d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Grid_sampler_2d_out"
    assert out.dtype == DType.FLOAT32


def test_grid_sampler_3d() -> None:
    """Verify grid_sampler_3d records a symbolic Grid_sampler_3d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.grid_sampler_3d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Grid_sampler_3d_out"
    assert out.dtype == DType.FLOAT32


def test_group_norm() -> None:
    """Verify group_norm records a symbolic Group_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.group_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Group_norm_out"
    assert out.dtype == DType.FLOAT32


def test_gru_cell() -> None:
    """Verify gru_cell records a symbolic Gru_cell tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.gru_cell(x)
    assert isinstance(out, Tensor)
    assert out.name == "Gru_cell_out"
    assert out.dtype == DType.FLOAT32


def test_gt() -> None:
    """Verify gt records a symbolic Gt tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.gt(x)
    assert isinstance(out, Tensor)
    assert out.name == "Gt_out"
    assert out.dtype == DType.FLOAT32


def test_hamming_window() -> None:
    """Verify hamming_window records a symbolic Hamming_window tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.hamming_window(x)
    assert isinstance(out, Tensor)
    assert out.name == "Hamming_window_out"
    assert out.dtype == DType.FLOAT32


def test_hann_window() -> None:
    """Verify hann_window records a symbolic Hann_window tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.hann_window(x)
    assert isinstance(out, Tensor)
    assert out.name == "Hann_window_out"
    assert out.dtype == DType.FLOAT32


def test_hardshrink() -> None:
    """Verify hardshrink records a symbolic Hardshrink tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.hardshrink(x)
    assert isinstance(out, Tensor)
    assert out.name == "Hardshrink_out"
    assert out.dtype == DType.FLOAT32


def test_hash_tensor() -> None:
    """Verify hash_tensor records a symbolic Hash_tensor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.hash_tensor(x)
    assert isinstance(out, Tensor)
    assert out.name == "Hash_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_heaviside() -> None:
    """Verify heaviside records a symbolic Heaviside tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.heaviside(x)
    assert isinstance(out, Tensor)
    assert out.name == "Heaviside_out"
    assert out.dtype == DType.FLOAT32


def test_hinge_embedding_loss() -> None:
    """Verify hinge_embedding_loss records a symbolic Hinge_embedding_loss tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.hinge_embedding_loss(x)
    assert isinstance(out, Tensor)
    assert out.name == "Hinge_embedding_loss_out"
    assert out.dtype == DType.FLOAT32


def test_histc() -> None:
    """Verify histc records a symbolic Histc tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.histc(x)
    assert isinstance(out, Tensor)
    assert out.name == "Histc_out"
    assert out.dtype == DType.FLOAT32


def test_histogram() -> None:
    """Verify histogram records a symbolic Histogram tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.histogram(x)
    assert isinstance(out, Tensor)
    assert out.name == "Histogram_out"
    assert out.dtype == DType.FLOAT32


def test_histogramdd() -> None:
    """Verify histogramdd records a symbolic Histogramdd tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.histogramdd(x)
    assert isinstance(out, Tensor)
    assert out.name == "Histogramdd_out"
    assert out.dtype == DType.FLOAT32


def test_hsmm() -> None:
    """Verify hsmm records a symbolic Hsmm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.hsmm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Hsmm_out"
    assert out.dtype == DType.FLOAT32


def test_hsplit() -> None:
    """Verify hsplit records a symbolic Hsplit tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.hsplit(x)
    assert isinstance(out, Tensor)
    assert out.name == "Hsplit_out"
    assert out.dtype == DType.FLOAT32


def test_hspmm() -> None:
    """Verify hspmm records a symbolic Hspmm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.hspmm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Hspmm_out"
    assert out.dtype == DType.FLOAT32


def test_hstack() -> None:
    """Verify hstack records a symbolic Hstack tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.hstack(x)
    assert isinstance(out, Tensor)
    assert out.name == "Hstack_out"
    assert out.dtype == DType.FLOAT32


def test_hypot() -> None:
    """Verify hypot records a symbolic Hypot tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.hypot(x)
    assert isinstance(out, Tensor)
    assert out.name == "Hypot_out"
    assert out.dtype == DType.FLOAT32


def test_i0() -> None:
    """Verify i0 records a symbolic I0 tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.i0(x)
    assert isinstance(out, Tensor)
    assert out.name == "I0_out"
    assert out.dtype == DType.FLOAT32


def test_i0_() -> None:
    """Verify i0_ records a symbolic I0_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.i0_(x)
    assert isinstance(out, Tensor)
    assert out.name == "I0__out"
    assert out.dtype == DType.FLOAT32


def test_igamma() -> None:
    """Verify igamma records a symbolic Igamma tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.igamma(x)
    assert isinstance(out, Tensor)
    assert out.name == "Igamma_out"
    assert out.dtype == DType.FLOAT32


def test_igammac() -> None:
    """Verify igammac records a symbolic Igammac tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.igammac(x)
    assert isinstance(out, Tensor)
    assert out.name == "Igammac_out"
    assert out.dtype == DType.FLOAT32


def test_imag() -> None:
    """Verify imag records a symbolic Imag tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.imag(x)
    assert isinstance(out, Tensor)
    assert out.name == "Imag_out"
    assert out.dtype == DType.FLOAT32


def test_index_add() -> None:
    """Verify index_add records a symbolic Index_add tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.index_add(x)
    assert isinstance(out, Tensor)
    assert out.name == "Index_add_out"
    assert out.dtype == DType.FLOAT32


def test_index_copy() -> None:
    """Verify index_copy records a symbolic Index_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.index_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Index_copy_out"
    assert out.dtype == DType.FLOAT32


def test_index_fill() -> None:
    """Verify index_fill records a symbolic Index_fill tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.index_fill(x)
    assert isinstance(out, Tensor)
    assert out.name == "Index_fill_out"
    assert out.dtype == DType.FLOAT32


def test_index_put() -> None:
    """Verify index_put records a symbolic Index_put tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.index_put(x)
    assert isinstance(out, Tensor)
    assert out.name == "Index_put_out"
    assert out.dtype == DType.FLOAT32


def test_index_put_() -> None:
    """Verify index_put_ records a symbolic Index_put_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.index_put_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Index_put__out"
    assert out.dtype == DType.FLOAT32


def test_index_reduce() -> None:
    """Verify index_reduce records a symbolic Index_reduce tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.index_reduce(x)
    assert isinstance(out, Tensor)
    assert out.name == "Index_reduce_out"
    assert out.dtype == DType.FLOAT32


def test_index_select() -> None:
    """Verify index_select records a symbolic Index_select tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.index_select(x)
    assert isinstance(out, Tensor)
    assert out.name == "Index_select_out"
    assert out.dtype == DType.FLOAT32


def test_indices_copy() -> None:
    """Verify indices_copy records a symbolic Indices_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.indices_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Indices_copy_out"
    assert out.dtype == DType.FLOAT32


def test_inner() -> None:
    """Verify inner records a symbolic Inner tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.inner(x)
    assert isinstance(out, Tensor)
    assert out.name == "Inner_out"
    assert out.dtype == DType.FLOAT32


def test_instance_norm() -> None:
    """Verify instance_norm records a symbolic Instance_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.instance_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Instance_norm_out"
    assert out.dtype == DType.FLOAT32


def test_int_repr() -> None:
    """Verify int_repr records a symbolic Int_repr tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.int_repr(x)
    assert isinstance(out, Tensor)
    assert out.name == "Int_repr_out"
    assert out.dtype == DType.FLOAT32


def test_inverse() -> None:
    """Verify inverse records a symbolic Inverse tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.inverse(x)
    assert isinstance(out, Tensor)
    assert out.name == "Inverse_out"
    assert out.dtype == DType.FLOAT32


def test_is_complex() -> None:
    """Verify is_complex records a symbolic Is_complex tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.is_complex(x)
    assert isinstance(out, Tensor)
    assert out.name == "Is_complex_out"
    assert out.dtype == DType.FLOAT32


def test_is_conj() -> None:
    """Verify is_conj records a symbolic Is_conj tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.is_conj(x)
    assert isinstance(out, Tensor)
    assert out.name == "Is_conj_out"
    assert out.dtype == DType.FLOAT32


def test_is_floating_point() -> None:
    """Verify is_floating_point records a symbolic Is_floating_point tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.is_floating_point(x)
    assert isinstance(out, Tensor)
    assert out.name == "Is_floating_point_out"
    assert out.dtype == DType.FLOAT32


def test_is_neg() -> None:
    """Verify is_neg records a symbolic Is_neg tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.is_neg(x)
    assert isinstance(out, Tensor)
    assert out.name == "Is_neg_out"
    assert out.dtype == DType.FLOAT32


def test_is_nonzero() -> None:
    """Verify is_nonzero records a symbolic Is_nonzero tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.is_nonzero(x)
    assert isinstance(out, Tensor)
    assert out.name == "Is_nonzero_out"
    assert out.dtype == DType.FLOAT32


def test_is_same_size() -> None:
    """Verify is_same_size records a symbolic Is_same_size tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.is_same_size(x)
    assert isinstance(out, Tensor)
    assert out.name == "Is_same_size_out"
    assert out.dtype == DType.FLOAT32


def test_is_signed() -> None:
    """Verify is_signed records a symbolic Is_signed tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.is_signed(x)
    assert isinstance(out, Tensor)
    assert out.name == "Is_signed_out"
    assert out.dtype == DType.FLOAT32


def test_isclose() -> None:
    """Verify isclose records a symbolic Isclose tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.isclose(x)
    assert isinstance(out, Tensor)
    assert out.name == "Isclose_out"
    assert out.dtype == DType.FLOAT32


def test_isfinite() -> None:
    """Verify isfinite records a symbolic Isfinite tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.isfinite(x)
    assert isinstance(out, Tensor)
    assert out.name == "Isfinite_out"
    assert out.dtype == DType.FLOAT32


def test_isin() -> None:
    """Verify isin records a symbolic Isin tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.isin(x)
    assert isinstance(out, Tensor)
    assert out.name == "Isin_out"
    assert out.dtype == DType.FLOAT32


def test_isneginf() -> None:
    """Verify isneginf records a symbolic Isneginf tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.isneginf(x)
    assert isinstance(out, Tensor)
    assert out.name == "Isneginf_out"
    assert out.dtype == DType.FLOAT32


def test_isposinf() -> None:
    """Verify isposinf records a symbolic Isposinf tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.isposinf(x)
    assert isinstance(out, Tensor)
    assert out.name == "Isposinf_out"
    assert out.dtype == DType.FLOAT32


def test_isreal() -> None:
    """Verify isreal records a symbolic Isreal tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.isreal(x)
    assert isinstance(out, Tensor)
    assert out.name == "Isreal_out"
    assert out.dtype == DType.FLOAT32


def test_istft() -> None:
    """Verify istft records a symbolic Istft tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.istft(x)
    assert isinstance(out, Tensor)
    assert out.name == "Istft_out"
    assert out.dtype == DType.FLOAT32


def test_kaiser_window() -> None:
    """Verify kaiser_window records a symbolic Kaiser_window tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.kaiser_window(x)
    assert isinstance(out, Tensor)
    assert out.name == "Kaiser_window_out"
    assert out.dtype == DType.FLOAT32


def test_kl_div() -> None:
    """Verify kl_div records a symbolic Kl_div tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.kl_div(x)
    assert isinstance(out, Tensor)
    assert out.name == "Kl_div_out"
    assert out.dtype == DType.FLOAT32


def test_kron() -> None:
    """Verify kron records a symbolic Kron tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.kron(x)
    assert isinstance(out, Tensor)
    assert out.name == "Kron_out"
    assert out.dtype == DType.FLOAT32


def test_kthvalue() -> None:
    """Verify kthvalue records a symbolic Kthvalue tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.kthvalue(x)
    assert isinstance(out, Tensor)
    assert out.name == "Kthvalue_out"
    assert out.dtype == DType.FLOAT32


def test_layer_norm() -> None:
    """Verify layer_norm records a symbolic Layer_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.layer_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Layer_norm_out"
    assert out.dtype == DType.FLOAT32


def test_lcm() -> None:
    """Verify lcm records a symbolic Lcm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.lcm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Lcm_out"
    assert out.dtype == DType.FLOAT32


def test_lcm_() -> None:
    """Verify lcm_ records a symbolic Lcm_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.lcm_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Lcm__out"
    assert out.dtype == DType.FLOAT32


def test_ldexp() -> None:
    """Verify ldexp records a symbolic Ldexp tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ldexp(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ldexp_out"
    assert out.dtype == DType.FLOAT32


def test_ldexp_() -> None:
    """Verify ldexp_ records a symbolic Ldexp_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ldexp_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ldexp__out"
    assert out.dtype == DType.FLOAT32


def test_le() -> None:
    """Verify le records a symbolic Le tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.le(x)
    assert isinstance(out, Tensor)
    assert out.name == "Le_out"
    assert out.dtype == DType.FLOAT32


def test_lerp() -> None:
    """Verify lerp records a symbolic Lerp tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.lerp(x)
    assert isinstance(out, Tensor)
    assert out.name == "Lerp_out"
    assert out.dtype == DType.FLOAT32


def test_less_equal() -> None:
    """Verify less_equal records a symbolic Less_equal tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.less_equal(x)
    assert isinstance(out, Tensor)
    assert out.name == "Less_equal_out"
    assert out.dtype == DType.FLOAT32


def test_lgamma() -> None:
    """Verify lgamma records a symbolic Lgamma tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.lgamma(x)
    assert isinstance(out, Tensor)
    assert out.name == "Lgamma_out"
    assert out.dtype == DType.FLOAT32


def test_linspace() -> None:
    """Verify linspace records a symbolic Linspace tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.linspace(x)
    assert isinstance(out, Tensor)
    assert out.name == "Linspace_out"
    assert out.dtype == DType.FLOAT32


def test_log() -> None:
    """Verify log records a symbolic Log tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.log(x)
    assert isinstance(out, Tensor)
    assert out.name == "Log_out"
    assert out.dtype == DType.FLOAT32


def test_log10() -> None:
    """Verify log10 records a symbolic Log10 tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.log10(x)
    assert isinstance(out, Tensor)
    assert out.name == "Log10_out"
    assert out.dtype == DType.FLOAT32


def test_log10_() -> None:
    """Verify log10_ records a symbolic Log10_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.log10_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Log10__out"
    assert out.dtype == DType.FLOAT32


def test_log1p() -> None:
    """Verify log1p records a symbolic Log1p tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.log1p(x)
    assert isinstance(out, Tensor)
    assert out.name == "Log1p_out"
    assert out.dtype == DType.FLOAT32


def test_log1p_() -> None:
    """Verify log1p_ records a symbolic Log1p_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.log1p_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Log1p__out"
    assert out.dtype == DType.FLOAT32


def test_log2() -> None:
    """Verify log2 records a symbolic Log2 tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.log2(x)
    assert isinstance(out, Tensor)
    assert out.name == "Log2_out"
    assert out.dtype == DType.FLOAT32


def test_log2_() -> None:
    """Verify log2_ records a symbolic Log2_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.log2_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Log2__out"
    assert out.dtype == DType.FLOAT32


def test_log_() -> None:
    """Verify log_ records a symbolic Log_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.log_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Log__out"
    assert out.dtype == DType.FLOAT32


def test_logaddexp() -> None:
    """Verify logaddexp records a symbolic Logaddexp tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logaddexp(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logaddexp_out"
    assert out.dtype == DType.FLOAT32


def test_logaddexp2() -> None:
    """Verify logaddexp2 records a symbolic Logaddexp2 tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logaddexp2(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logaddexp2_out"
    assert out.dtype == DType.FLOAT32


def test_logcumsumexp() -> None:
    """Verify logcumsumexp records a symbolic Logcumsumexp tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logcumsumexp(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logcumsumexp_out"
    assert out.dtype == DType.FLOAT32


def test_logdet() -> None:
    """Verify logdet records a symbolic Logdet tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logdet(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logdet_out"
    assert out.dtype == DType.FLOAT32


def test_logical_and() -> None:
    """Verify logical_and records a symbolic Logical_and tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logical_and(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logical_and_out"
    assert out.dtype == DType.FLOAT32


def test_logical_not() -> None:
    """Verify logical_not records a symbolic Logical_not tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logical_not(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logical_not_out"
    assert out.dtype == DType.FLOAT32


def test_logical_or() -> None:
    """Verify logical_or records a symbolic Logical_or tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logical_or(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logical_or_out"
    assert out.dtype == DType.FLOAT32


def test_logical_xor() -> None:
    """Verify logical_xor records a symbolic Logical_xor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logical_xor(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logical_xor_out"
    assert out.dtype == DType.FLOAT32


def test_logit() -> None:
    """Verify logit records a symbolic Logit tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logit(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logit_out"
    assert out.dtype == DType.FLOAT32


def test_logit_() -> None:
    """Verify logit_ records a symbolic Logit_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logit_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logit__out"
    assert out.dtype == DType.FLOAT32


def test_logspace() -> None:
    """Verify logspace records a symbolic Logspace tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logspace(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logspace_out"
    assert out.dtype == DType.FLOAT32


def test_logsumexp() -> None:
    """Verify logsumexp records a symbolic Logsumexp tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.logsumexp(x)
    assert isinstance(out, Tensor)
    assert out.name == "Logsumexp_out"
    assert out.dtype == DType.FLOAT32


def test_lstm_cell() -> None:
    """Verify lstm_cell records a symbolic Lstm_cell tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.lstm_cell(x)
    assert isinstance(out, Tensor)
    assert out.name == "Lstm_cell_out"
    assert out.dtype == DType.FLOAT32


def test_lt() -> None:
    """Verify lt records a symbolic Lt tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.lt(x)
    assert isinstance(out, Tensor)
    assert out.name == "Lt_out"
    assert out.dtype == DType.FLOAT32


def test_lu_solve() -> None:
    """Verify lu_solve records a symbolic Lu_solve tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.lu_solve(x)
    assert isinstance(out, Tensor)
    assert out.name == "Lu_solve_out"
    assert out.dtype == DType.FLOAT32


def test_lu_unpack() -> None:
    """Verify lu_unpack records a symbolic Lu_unpack tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.lu_unpack(x)
    assert isinstance(out, Tensor)
    assert out.name == "Lu_unpack_out"
    assert out.dtype == DType.FLOAT32


def test_margin_ranking_loss() -> None:
    """Verify margin_ranking_loss records a symbolic Margin_ranking_loss tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.margin_ranking_loss(x)
    assert isinstance(out, Tensor)
    assert out.name == "Margin_ranking_loss_out"
    assert out.dtype == DType.FLOAT32


def test_masked_fill() -> None:
    """Verify masked_fill records a symbolic Masked_fill tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.masked_fill(x)
    assert isinstance(out, Tensor)
    assert out.name == "Masked_fill_out"
    assert out.dtype == DType.FLOAT32


def test_masked_scatter() -> None:
    """Verify masked_scatter records a symbolic Masked_scatter tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.masked_scatter(x)
    assert isinstance(out, Tensor)
    assert out.name == "Masked_scatter_out"
    assert out.dtype == DType.FLOAT32


def test_masked_select() -> None:
    """Verify masked_select records a symbolic Masked_select tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.masked_select(x)
    assert isinstance(out, Tensor)
    assert out.name == "Masked_select_out"
    assert out.dtype == DType.FLOAT32


def test_matrix_exp() -> None:
    """Verify matrix_exp records a symbolic Matrix_exp tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.matrix_exp(x)
    assert isinstance(out, Tensor)
    assert out.name == "Matrix_exp_out"
    assert out.dtype == DType.FLOAT32


def test_matrix_power() -> None:
    """Verify matrix_power records a symbolic Matrix_power tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.matrix_power(x)
    assert isinstance(out, Tensor)
    assert out.name == "Matrix_power_out"
    assert out.dtype == DType.FLOAT32


def test_max_pool1d() -> None:
    """Verify max_pool1d records a symbolic Max_pool1d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.max_pool1d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Max_pool1d_out"
    assert out.dtype == DType.FLOAT32


def test_max_pool1d_with_indices() -> None:
    """Verify max_pool1d_with_indices records a symbolic Max_pool1d_with_indices tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.max_pool1d_with_indices(x)
    assert isinstance(out, Tensor)
    assert out.name == "Max_pool1d_with_indices_out"
    assert out.dtype == DType.FLOAT32


def test_max_pool2d() -> None:
    """Verify max_pool2d records a symbolic Max_pool2d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.max_pool2d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Max_pool2d_out"
    assert out.dtype == DType.FLOAT32


def test_max_pool3d() -> None:
    """Verify max_pool3d records a symbolic Max_pool3d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.max_pool3d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Max_pool3d_out"
    assert out.dtype == DType.FLOAT32


def test_maximum() -> None:
    """Verify maximum records a symbolic Maximum tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.maximum(x)
    assert isinstance(out, Tensor)
    assert out.name == "Maximum_out"
    assert out.dtype == DType.FLOAT32


def test_median() -> None:
    """Verify median records a symbolic Median tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.median(x)
    assert isinstance(out, Tensor)
    assert out.name == "Median_out"
    assert out.dtype == DType.FLOAT32


def test_meshgrid() -> None:
    """Verify meshgrid records a symbolic Meshgrid tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.meshgrid(x)
    assert isinstance(out, Tensor)
    assert out.name == "Meshgrid_out"
    assert out.dtype == DType.FLOAT32


def test_minimum() -> None:
    """Verify minimum records a symbolic Minimum tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.minimum(x)
    assert isinstance(out, Tensor)
    assert out.name == "Minimum_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_batch_norm() -> None:
    """Verify miopen_batch_norm records a symbolic Miopen_batch_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.miopen_batch_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Miopen_batch_norm_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_convolution() -> None:
    """Verify miopen_convolution records a symbolic Miopen_convolution tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.miopen_convolution(x)
    assert isinstance(out, Tensor)
    assert out.name == "Miopen_convolution_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_convolution_add_relu() -> None:
    """Verify miopen_convolution_add_relu records a symbolic Miopen_convolution_add_relu tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.miopen_convolution_add_relu(x)
    assert isinstance(out, Tensor)
    assert out.name == "Miopen_convolution_add_relu_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_convolution_relu() -> None:
    """Verify miopen_convolution_relu records a symbolic Miopen_convolution_relu tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.miopen_convolution_relu(x)
    assert isinstance(out, Tensor)
    assert out.name == "Miopen_convolution_relu_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_convolution_transpose() -> None:
    """Verify miopen_convolution_transpose records a symbolic Miopen_convolution_transpose tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.miopen_convolution_transpose(x)
    assert isinstance(out, Tensor)
    assert out.name == "Miopen_convolution_transpose_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_ctc_loss() -> None:
    """Verify miopen_ctc_loss records a symbolic Miopen_ctc_loss tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.miopen_ctc_loss(x)
    assert isinstance(out, Tensor)
    assert out.name == "Miopen_ctc_loss_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_depthwise_convolution() -> None:
    """Verify miopen_depthwise_convolution records a symbolic Miopen_depthwise_convolution tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.miopen_depthwise_convolution(x)
    assert isinstance(out, Tensor)
    assert out.name == "Miopen_depthwise_convolution_out"
    assert out.dtype == DType.FLOAT32


def test_miopen_rnn() -> None:
    """Verify miopen_rnn records a symbolic Miopen_rnn tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.miopen_rnn(x)
    assert isinstance(out, Tensor)
    assert out.name == "Miopen_rnn_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_adaptive_avg_pool2d() -> None:
    """Verify mkldnn_adaptive_avg_pool2d records a symbolic Mkldnn_adaptive_avg_pool2d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.mkldnn_adaptive_avg_pool2d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Mkldnn_adaptive_avg_pool2d_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_convolution() -> None:
    """Verify mkldnn_convolution records a symbolic Mkldnn_convolution tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.mkldnn_convolution(x)
    assert isinstance(out, Tensor)
    assert out.name == "Mkldnn_convolution_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_linear_backward_weights() -> None:
    """Verify mkldnn_linear_backward_weights records a symbolic Mkldnn_linear_backward_weights tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.mkldnn_linear_backward_weights(x)
    assert isinstance(out, Tensor)
    assert out.name == "Mkldnn_linear_backward_weights_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_max_pool2d() -> None:
    """Verify mkldnn_max_pool2d records a symbolic Mkldnn_max_pool2d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.mkldnn_max_pool2d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Mkldnn_max_pool2d_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_max_pool3d() -> None:
    """Verify mkldnn_max_pool3d records a symbolic Mkldnn_max_pool3d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.mkldnn_max_pool3d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Mkldnn_max_pool3d_out"
    assert out.dtype == DType.FLOAT32


def test_mkldnn_rnn_layer() -> None:
    """Verify mkldnn_rnn_layer records a symbolic Mkldnn_rnn_layer tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.mkldnn_rnn_layer(x)
    assert isinstance(out, Tensor)
    assert out.name == "Mkldnn_rnn_layer_out"
    assert out.dtype == DType.FLOAT32


def test_mm() -> None:
    """Verify mm records a symbolic Mm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.mm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Mm_out"
    assert out.dtype == DType.FLOAT32


def test_mode() -> None:
    """Verify mode records a symbolic Mode tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.mode(x)
    assert isinstance(out, Tensor)
    assert out.name == "Mode_out"
    assert out.dtype == DType.FLOAT32


def test_moveaxis() -> None:
    """Verify moveaxis records a symbolic Moveaxis tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.moveaxis(x)
    assert isinstance(out, Tensor)
    assert out.name == "Moveaxis_out"
    assert out.dtype == DType.FLOAT32


def test_movedim() -> None:
    """Verify movedim records a symbolic Movedim tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.movedim(x)
    assert isinstance(out, Tensor)
    assert out.name == "Movedim_out"
    assert out.dtype == DType.FLOAT32


def test_msort() -> None:
    """Verify msort records a symbolic Msort tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.msort(x)
    assert isinstance(out, Tensor)
    assert out.name == "Msort_out"
    assert out.dtype == DType.FLOAT32


def test_multiply() -> None:
    """Verify multiply records a symbolic Multiply tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.multiply(x)
    assert isinstance(out, Tensor)
    assert out.name == "Multiply_out"
    assert out.dtype == DType.FLOAT32


def test_mv() -> None:
    """Verify mv records a symbolic Mv tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.mv(x)
    assert isinstance(out, Tensor)
    assert out.name == "Mv_out"
    assert out.dtype == DType.FLOAT32


def test_mvlgamma() -> None:
    """Verify mvlgamma records a symbolic Mvlgamma tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.mvlgamma(x)
    assert isinstance(out, Tensor)
    assert out.name == "Mvlgamma_out"
    assert out.dtype == DType.FLOAT32


def test_nan_to_num() -> None:
    """Verify nan_to_num records a symbolic Nan_to_num tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.nan_to_num(x)
    assert isinstance(out, Tensor)
    assert out.name == "Nan_to_num_out"
    assert out.dtype == DType.FLOAT32


def test_nan_to_num_() -> None:
    """Verify nan_to_num_ records a symbolic Nan_to_num_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.nan_to_num_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Nan_to_num__out"
    assert out.dtype == DType.FLOAT32


def test_nanmean() -> None:
    """Verify nanmean records a symbolic Nanmean tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.nanmean(x)
    assert isinstance(out, Tensor)
    assert out.name == "Nanmean_out"
    assert out.dtype == DType.FLOAT32


def test_nanmedian() -> None:
    """Verify nanmedian records a symbolic Nanmedian tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.nanmedian(x)
    assert isinstance(out, Tensor)
    assert out.name == "Nanmedian_out"
    assert out.dtype == DType.FLOAT32


def test_nanquantile() -> None:
    """Verify nanquantile records a symbolic Nanquantile tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.nanquantile(x)
    assert isinstance(out, Tensor)
    assert out.name == "Nanquantile_out"
    assert out.dtype == DType.FLOAT32


def test_nansum() -> None:
    """Verify nansum records a symbolic Nansum tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.nansum(x)
    assert isinstance(out, Tensor)
    assert out.name == "Nansum_out"
    assert out.dtype == DType.FLOAT32


def test_narrow() -> None:
    """Verify narrow records a symbolic Narrow tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.narrow(x)
    assert isinstance(out, Tensor)
    assert out.name == "Narrow_out"
    assert out.dtype == DType.FLOAT32


def test_narrow_copy() -> None:
    """Verify narrow_copy records a symbolic Narrow_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.narrow_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Narrow_copy_out"
    assert out.dtype == DType.FLOAT32


def test_native_batch_norm() -> None:
    """Verify native_batch_norm records a symbolic Native_batch_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.native_batch_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Native_batch_norm_out"
    assert out.dtype == DType.FLOAT32


def test_native_channel_shuffle() -> None:
    """Verify native_channel_shuffle records a symbolic Native_channel_shuffle tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.native_channel_shuffle(x)
    assert isinstance(out, Tensor)
    assert out.name == "Native_channel_shuffle_out"
    assert out.dtype == DType.FLOAT32


def test_native_dropout() -> None:
    """Verify native_dropout records a symbolic Native_dropout tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.native_dropout(x)
    assert isinstance(out, Tensor)
    assert out.name == "Native_dropout_out"
    assert out.dtype == DType.FLOAT32


def test_native_group_norm() -> None:
    """Verify native_group_norm records a symbolic Native_group_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.native_group_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Native_group_norm_out"
    assert out.dtype == DType.FLOAT32


def test_native_layer_norm() -> None:
    """Verify native_layer_norm records a symbolic Native_layer_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.native_layer_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Native_layer_norm_out"
    assert out.dtype == DType.FLOAT32


def test_native_norm() -> None:
    """Verify native_norm records a symbolic Native_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.native_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Native_norm_out"
    assert out.dtype == DType.FLOAT32


def test_ne() -> None:
    """Verify ne records a symbolic Ne tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ne(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ne_out"
    assert out.dtype == DType.FLOAT32


def test_neg_() -> None:
    """Verify neg_ records a symbolic Neg_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.neg_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Neg__out"
    assert out.dtype == DType.FLOAT32


def test_negative() -> None:
    """Verify negative records a symbolic Negative tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.negative(x)
    assert isinstance(out, Tensor)
    assert out.name == "Negative_out"
    assert out.dtype == DType.FLOAT32


def test_negative_() -> None:
    """Verify negative_ records a symbolic Negative_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.negative_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Negative__out"
    assert out.dtype == DType.FLOAT32


def test_nextafter() -> None:
    """Verify nextafter records a symbolic Nextafter tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.nextafter(x)
    assert isinstance(out, Tensor)
    assert out.name == "Nextafter_out"
    assert out.dtype == DType.FLOAT32


def test_nonzero() -> None:
    """Verify nonzero records a symbolic Nonzero tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.nonzero(x)
    assert isinstance(out, Tensor)
    assert out.name == "Nonzero_out"
    assert out.dtype == DType.FLOAT32


def test_nonzero_static() -> None:
    """Verify nonzero_static records a symbolic Nonzero_static tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.nonzero_static(x)
    assert isinstance(out, Tensor)
    assert out.name == "Nonzero_static_out"
    assert out.dtype == DType.FLOAT32


def test_norm() -> None:
    """Verify norm records a symbolic Norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Norm_out"
    assert out.dtype == DType.FLOAT32


def test_norm_except_dim() -> None:
    """Verify norm_except_dim records a symbolic Norm_except_dim tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.norm_except_dim(x)
    assert isinstance(out, Tensor)
    assert out.name == "Norm_except_dim_out"
    assert out.dtype == DType.FLOAT32


def test_normal() -> None:
    """Verify normal records a symbolic Normal tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.normal(x)
    assert isinstance(out, Tensor)
    assert out.name == "Normal_out"
    assert out.dtype == DType.FLOAT32


def test_not_equal() -> None:
    """Verify not_equal records a symbolic Not_equal tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.not_equal(x)
    assert isinstance(out, Tensor)
    assert out.name == "Not_equal_out"
    assert out.dtype == DType.FLOAT32


def test_nuclear_norm() -> None:
    """Verify nuclear_norm records a symbolic Nuclear_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.nuclear_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Nuclear_norm_out"
    assert out.dtype == DType.FLOAT32


def test_numel() -> None:
    """Verify numel records a symbolic Numel tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.numel(x)
    assert isinstance(out, Tensor)
    assert out.name == "Numel_out"
    assert out.dtype == DType.FLOAT32


def test_ones_like() -> None:
    """Verify ones_like records a symbolic Ones_like tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ones_like(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ones_like_out"
    assert out.dtype == DType.FLOAT32


def test_orgqr() -> None:
    """Verify orgqr records a symbolic Orgqr tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.orgqr(x)
    assert isinstance(out, Tensor)
    assert out.name == "Orgqr_out"
    assert out.dtype == DType.FLOAT32


def test_ormqr() -> None:
    """Verify ormqr records a symbolic Ormqr tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ormqr(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ormqr_out"
    assert out.dtype == DType.FLOAT32


def test_outer() -> None:
    """Verify outer records a symbolic Outer tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.outer(x)
    assert isinstance(out, Tensor)
    assert out.name == "Outer_out"
    assert out.dtype == DType.FLOAT32


def test_pairwise_distance() -> None:
    """Verify pairwise_distance records a symbolic Pairwise_distance tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.pairwise_distance(x)
    assert isinstance(out, Tensor)
    assert out.name == "Pairwise_distance_out"
    assert out.dtype == DType.FLOAT32


def test_pdist() -> None:
    """Verify pdist records a symbolic Pdist tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.pdist(x)
    assert isinstance(out, Tensor)
    assert out.name == "Pdist_out"
    assert out.dtype == DType.FLOAT32


def test_permute() -> None:
    """Verify permute records a symbolic Permute tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.permute(x)
    assert isinstance(out, Tensor)
    assert out.name == "Permute_out"
    assert out.dtype == DType.FLOAT32


def test_permute_copy() -> None:
    """Verify permute_copy records a symbolic Permute_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.permute_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Permute_copy_out"
    assert out.dtype == DType.FLOAT32


def test_pinverse() -> None:
    """Verify pinverse records a symbolic Pinverse tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.pinverse(x)
    assert isinstance(out, Tensor)
    assert out.name == "Pinverse_out"
    assert out.dtype == DType.FLOAT32


def test_pixel_shuffle() -> None:
    """Verify pixel_shuffle records a symbolic Pixel_shuffle tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.pixel_shuffle(x)
    assert isinstance(out, Tensor)
    assert out.name == "Pixel_shuffle_out"
    assert out.dtype == DType.FLOAT32


def test_pixel_unshuffle() -> None:
    """Verify pixel_unshuffle records a symbolic Pixel_unshuffle tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.pixel_unshuffle(x)
    assert isinstance(out, Tensor)
    assert out.name == "Pixel_unshuffle_out"
    assert out.dtype == DType.FLOAT32


def test_poisson() -> None:
    """Verify poisson records a symbolic Poisson tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.poisson(x)
    assert isinstance(out, Tensor)
    assert out.name == "Poisson_out"
    assert out.dtype == DType.FLOAT32


def test_poisson_nll_loss() -> None:
    """Verify poisson_nll_loss records a symbolic Poisson_nll_loss tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.poisson_nll_loss(x)
    assert isinstance(out, Tensor)
    assert out.name == "Poisson_nll_loss_out"
    assert out.dtype == DType.FLOAT32


def test_polar() -> None:
    """Verify polar records a symbolic Polar tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.polar(x)
    assert isinstance(out, Tensor)
    assert out.name == "Polar_out"
    assert out.dtype == DType.FLOAT32


def test_polygamma() -> None:
    """Verify polygamma records a symbolic Polygamma tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.polygamma(x)
    assert isinstance(out, Tensor)
    assert out.name == "Polygamma_out"
    assert out.dtype == DType.FLOAT32


def test_positive() -> None:
    """Verify positive records a symbolic Positive tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.positive(x)
    assert isinstance(out, Tensor)
    assert out.name == "Positive_out"
    assert out.dtype == DType.FLOAT32


def test_prod() -> None:
    """Verify prod records a symbolic Prod tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.prod(x)
    assert isinstance(out, Tensor)
    assert out.name == "Prod_out"
    assert out.dtype == DType.FLOAT32


def test_put() -> None:
    """Verify put records a symbolic Put tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.put(x)
    assert isinstance(out, Tensor)
    assert out.name == "Put_out"
    assert out.dtype == DType.FLOAT32


def test_q_per_channel_axis() -> None:
    """Verify q_per_channel_axis records a symbolic Q_per_channel_axis tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.q_per_channel_axis(x)
    assert isinstance(out, Tensor)
    assert out.name == "Q_per_channel_axis_out"
    assert out.dtype == DType.FLOAT32


def test_q_per_channel_scales() -> None:
    """Verify q_per_channel_scales records a symbolic Q_per_channel_scales tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.q_per_channel_scales(x)
    assert isinstance(out, Tensor)
    assert out.name == "Q_per_channel_scales_out"
    assert out.dtype == DType.FLOAT32


def test_q_per_channel_zero_points() -> None:
    """Verify q_per_channel_zero_points records a symbolic Q_per_channel_zero_points tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.q_per_channel_zero_points(x)
    assert isinstance(out, Tensor)
    assert out.name == "Q_per_channel_zero_points_out"
    assert out.dtype == DType.FLOAT32


def test_q_scale() -> None:
    """Verify q_scale records a symbolic Q_scale tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.q_scale(x)
    assert isinstance(out, Tensor)
    assert out.name == "Q_scale_out"
    assert out.dtype == DType.FLOAT32


def test_q_zero_point() -> None:
    """Verify q_zero_point records a symbolic Q_zero_point tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.q_zero_point(x)
    assert isinstance(out, Tensor)
    assert out.name == "Q_zero_point_out"
    assert out.dtype == DType.FLOAT32


def test_qr() -> None:
    """Verify qr records a symbolic Qr tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.qr(x)
    assert isinstance(out, Tensor)
    assert out.name == "Qr_out"
    assert out.dtype == DType.FLOAT32


def test_quantile() -> None:
    """Verify quantile records a symbolic Quantile tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantile(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantile_out"
    assert out.dtype == DType.FLOAT32


def test_quantize_per_channel() -> None:
    """Verify quantize_per_channel records a symbolic Quantize_per_channel tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantize_per_channel(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantize_per_channel_out"
    assert out.dtype == DType.FLOAT32


def test_quantize_per_tensor() -> None:
    """Verify quantize_per_tensor records a symbolic Quantize_per_tensor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantize_per_tensor(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantize_per_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_quantize_per_tensor_dynamic() -> None:
    """Verify quantize_per_tensor_dynamic records a symbolic Quantize_per_tensor_dynamic tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantize_per_tensor_dynamic(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantize_per_tensor_dynamic_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_batch_norm() -> None:
    """Verify quantized_batch_norm records a symbolic Quantized_batch_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantized_batch_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantized_batch_norm_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_gru_cell() -> None:
    """Verify quantized_gru_cell records a symbolic Quantized_gru_cell tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantized_gru_cell(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantized_gru_cell_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_lstm_cell() -> None:
    """Verify quantized_lstm_cell records a symbolic Quantized_lstm_cell tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantized_lstm_cell(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantized_lstm_cell_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_max_pool1d() -> None:
    """Verify quantized_max_pool1d records a symbolic Quantized_max_pool1d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantized_max_pool1d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantized_max_pool1d_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_max_pool2d() -> None:
    """Verify quantized_max_pool2d records a symbolic Quantized_max_pool2d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantized_max_pool2d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantized_max_pool2d_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_max_pool3d() -> None:
    """Verify quantized_max_pool3d records a symbolic Quantized_max_pool3d tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantized_max_pool3d(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantized_max_pool3d_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_rnn_relu_cell() -> None:
    """Verify quantized_rnn_relu_cell records a symbolic Quantized_rnn_relu_cell tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantized_rnn_relu_cell(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantized_rnn_relu_cell_out"
    assert out.dtype == DType.FLOAT32


def test_quantized_rnn_tanh_cell() -> None:
    """Verify quantized_rnn_tanh_cell records a symbolic Quantized_rnn_tanh_cell tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.quantized_rnn_tanh_cell(x)
    assert isinstance(out, Tensor)
    assert out.name == "Quantized_rnn_tanh_cell_out"
    assert out.dtype == DType.FLOAT32


def test_rad2deg() -> None:
    """Verify rad2deg records a symbolic Rad2deg tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rad2deg(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rad2deg_out"
    assert out.dtype == DType.FLOAT32


def test_rad2deg_() -> None:
    """Verify rad2deg_ records a symbolic Rad2deg_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rad2deg_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rad2deg__out"
    assert out.dtype == DType.FLOAT32


def test_rand_like() -> None:
    """Verify rand_like records a symbolic Rand_like tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rand_like(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rand_like_out"
    assert out.dtype == DType.FLOAT32


def test_randint() -> None:
    """Verify randint records a symbolic Randint tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.randint(x)
    assert isinstance(out, Tensor)
    assert out.name == "Randint_out"
    assert out.dtype == DType.FLOAT32


def test_randint_like() -> None:
    """Verify randint_like records a symbolic Randint_like tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.randint_like(x)
    assert isinstance(out, Tensor)
    assert out.name == "Randint_like_out"
    assert out.dtype == DType.FLOAT32


def test_randn_like() -> None:
    """Verify randn_like records a symbolic Randn_like tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.randn_like(x)
    assert isinstance(out, Tensor)
    assert out.name == "Randn_like_out"
    assert out.dtype == DType.FLOAT32


def test_randperm() -> None:
    """Verify randperm records a symbolic Randperm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.randperm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Randperm_out"
    assert out.dtype == DType.FLOAT32


def test_range() -> None:
    """Verify range records a symbolic Range tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.range(x)
    assert isinstance(out, Tensor)
    assert out.name == "Range_out"
    assert out.dtype == DType.FLOAT32


def test_ravel() -> None:
    """Verify ravel records a symbolic Ravel tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.ravel(x)
    assert isinstance(out, Tensor)
    assert out.name == "Ravel_out"
    assert out.dtype == DType.FLOAT32


def test_real() -> None:
    """Verify real records a symbolic Real tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.real(x)
    assert isinstance(out, Tensor)
    assert out.name == "Real_out"
    assert out.dtype == DType.FLOAT32


def test_reciprocal_() -> None:
    """Verify reciprocal_ records a symbolic Reciprocal_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.reciprocal_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Reciprocal__out"
    assert out.dtype == DType.FLOAT32


def test_relu_() -> None:
    """Verify relu_ records a symbolic Relu_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.relu_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Relu__out"
    assert out.dtype == DType.FLOAT32


def test_remainder() -> None:
    """Verify remainder records a symbolic Remainder tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.remainder(x)
    assert isinstance(out, Tensor)
    assert out.name == "Remainder_out"
    assert out.dtype == DType.FLOAT32


def test_renorm() -> None:
    """Verify renorm records a symbolic Renorm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.renorm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Renorm_out"
    assert out.dtype == DType.FLOAT32


def test_repeat_interleave() -> None:
    """Verify repeat_interleave records a symbolic Repeat_interleave tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.repeat_interleave(x)
    assert isinstance(out, Tensor)
    assert out.name == "Repeat_interleave_out"
    assert out.dtype == DType.FLOAT32


def test_resize_as_() -> None:
    """Verify resize_as_ records a symbolic Resize_as_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.resize_as_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Resize_as__out"
    assert out.dtype == DType.FLOAT32


def test_resize_as_sparse_() -> None:
    """Verify resize_as_sparse_ records a symbolic Resize_as_sparse_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.resize_as_sparse_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Resize_as_sparse__out"
    assert out.dtype == DType.FLOAT32


def test_resolve_conj() -> None:
    """Verify resolve_conj records a symbolic Resolve_conj tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.resolve_conj(x)
    assert isinstance(out, Tensor)
    assert out.name == "Resolve_conj_out"
    assert out.dtype == DType.FLOAT32


def test_resolve_neg() -> None:
    """Verify resolve_neg records a symbolic Resolve_neg tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.resolve_neg(x)
    assert isinstance(out, Tensor)
    assert out.name == "Resolve_neg_out"
    assert out.dtype == DType.FLOAT32


def test_rms_norm() -> None:
    """Verify rms_norm records a symbolic Rms_norm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rms_norm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rms_norm_out"
    assert out.dtype == DType.FLOAT32


def test_rnn_relu() -> None:
    """Verify rnn_relu records a symbolic Rnn_relu tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rnn_relu(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rnn_relu_out"
    assert out.dtype == DType.FLOAT32


def test_rnn_relu_cell() -> None:
    """Verify rnn_relu_cell records a symbolic Rnn_relu_cell tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rnn_relu_cell(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rnn_relu_cell_out"
    assert out.dtype == DType.FLOAT32


def test_rnn_tanh() -> None:
    """Verify rnn_tanh records a symbolic Rnn_tanh tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rnn_tanh(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rnn_tanh_out"
    assert out.dtype == DType.FLOAT32


def test_rnn_tanh_cell() -> None:
    """Verify rnn_tanh_cell records a symbolic Rnn_tanh_cell tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rnn_tanh_cell(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rnn_tanh_cell_out"
    assert out.dtype == DType.FLOAT32


def test_roll() -> None:
    """Verify roll records a symbolic Roll tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.roll(x)
    assert isinstance(out, Tensor)
    assert out.name == "Roll_out"
    assert out.dtype == DType.FLOAT32


def test_rot90() -> None:
    """Verify rot90 records a symbolic Rot90 tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rot90(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rot90_out"
    assert out.dtype == DType.FLOAT32


def test_round_() -> None:
    """Verify round_ records a symbolic Round_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.round_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Round__out"
    assert out.dtype == DType.FLOAT32


def test_row_indices_copy() -> None:
    """Verify row_indices_copy records a symbolic Row_indices_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.row_indices_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Row_indices_copy_out"
    assert out.dtype == DType.FLOAT32


def test_row_stack() -> None:
    """Verify row_stack records a symbolic Row_stack tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.row_stack(x)
    assert isinstance(out, Tensor)
    assert out.name == "Row_stack_out"
    assert out.dtype == DType.FLOAT32


def test_rrelu() -> None:
    """Verify rrelu records a symbolic Rrelu tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rrelu(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rrelu_out"
    assert out.dtype == DType.FLOAT32


def test_rrelu_() -> None:
    """Verify rrelu_ records a symbolic Rrelu_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rrelu_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rrelu__out"
    assert out.dtype == DType.FLOAT32


def test_rsqrt() -> None:
    """Verify rsqrt records a symbolic Rsqrt tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rsqrt(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rsqrt_out"
    assert out.dtype == DType.FLOAT32


def test_rsqrt_() -> None:
    """Verify rsqrt_ records a symbolic Rsqrt_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rsqrt_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rsqrt__out"
    assert out.dtype == DType.FLOAT32


def test_rsub() -> None:
    """Verify rsub records a symbolic Rsub tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.rsub(x)
    assert isinstance(out, Tensor)
    assert out.name == "Rsub_out"
    assert out.dtype == DType.FLOAT32


def test_saddmm() -> None:
    """Verify saddmm records a symbolic Saddmm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.saddmm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Saddmm_out"
    assert out.dtype == DType.FLOAT32


def test_scalar_tensor() -> None:
    """Verify scalar_tensor records a symbolic Scalar_tensor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.scalar_tensor(x)
    assert isinstance(out, Tensor)
    assert out.name == "Scalar_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_scatter_add() -> None:
    """Verify scatter_add records a symbolic Scatter_add tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.scatter_add(x)
    assert isinstance(out, Tensor)
    assert out.name == "Scatter_add_out"
    assert out.dtype == DType.FLOAT32


def test_scatter_reduce() -> None:
    """Verify scatter_reduce records a symbolic Scatter_reduce tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.scatter_reduce(x)
    assert isinstance(out, Tensor)
    assert out.name == "Scatter_reduce_out"
    assert out.dtype == DType.FLOAT32


def test_searchsorted() -> None:
    """Verify searchsorted records a symbolic Searchsorted tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.searchsorted(x)
    assert isinstance(out, Tensor)
    assert out.name == "Searchsorted_out"
    assert out.dtype == DType.FLOAT32


def test_select() -> None:
    """Verify select records a symbolic Select tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.select(x)
    assert isinstance(out, Tensor)
    assert out.name == "Select_out"
    assert out.dtype == DType.FLOAT32


def test_select_copy() -> None:
    """Verify select_copy records a symbolic Select_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.select_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Select_copy_out"
    assert out.dtype == DType.FLOAT32


def test_select_scatter() -> None:
    """Verify select_scatter records a symbolic Select_scatter tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.select_scatter(x)
    assert isinstance(out, Tensor)
    assert out.name == "Select_scatter_out"
    assert out.dtype == DType.FLOAT32


def test_selu_() -> None:
    """Verify selu_ records a symbolic Selu_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.selu_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Selu__out"
    assert out.dtype == DType.FLOAT32


def test_sgn() -> None:
    """Verify sgn records a symbolic Sgn tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sgn(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sgn_out"
    assert out.dtype == DType.FLOAT32


def test_sigmoid_() -> None:
    """Verify sigmoid_ records a symbolic Sigmoid_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sigmoid_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sigmoid__out"
    assert out.dtype == DType.FLOAT32


def test_signbit() -> None:
    """Verify signbit records a symbolic Signbit tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.signbit(x)
    assert isinstance(out, Tensor)
    assert out.name == "Signbit_out"
    assert out.dtype == DType.FLOAT32


def test_sin_() -> None:
    """Verify sin_ records a symbolic Sin_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sin_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sin__out"
    assert out.dtype == DType.FLOAT32


def test_sinc() -> None:
    """Verify sinc records a symbolic Sinc tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sinc(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sinc_out"
    assert out.dtype == DType.FLOAT32


def test_sinc_() -> None:
    """Verify sinc_ records a symbolic Sinc_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sinc_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sinc__out"
    assert out.dtype == DType.FLOAT32


def test_sinh_() -> None:
    """Verify sinh_ records a symbolic Sinh_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sinh_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sinh__out"
    assert out.dtype == DType.FLOAT32


def test_slice_copy() -> None:
    """Verify slice_copy records a symbolic Slice_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.slice_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Slice_copy_out"
    assert out.dtype == DType.FLOAT32


def test_slice_inverse() -> None:
    """Verify slice_inverse records a symbolic Slice_inverse tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.slice_inverse(x)
    assert isinstance(out, Tensor)
    assert out.name == "Slice_inverse_out"
    assert out.dtype == DType.FLOAT32


def test_slice_scatter() -> None:
    """Verify slice_scatter records a symbolic Slice_scatter tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.slice_scatter(x)
    assert isinstance(out, Tensor)
    assert out.name == "Slice_scatter_out"
    assert out.dtype == DType.FLOAT32


def test_slogdet() -> None:
    """Verify slogdet records a symbolic Slogdet tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.slogdet(x)
    assert isinstance(out, Tensor)
    assert out.name == "Slogdet_out"
    assert out.dtype == DType.FLOAT32


def test_smm() -> None:
    """Verify smm records a symbolic Smm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.smm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Smm_out"
    assert out.dtype == DType.FLOAT32


def test_sort() -> None:
    """Verify sort records a symbolic Sort tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sort(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sort_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_bsc_tensor() -> None:
    """Verify sparse_bsc_tensor records a symbolic Sparse_bsc_tensor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sparse_bsc_tensor(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sparse_bsc_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_bsr_tensor() -> None:
    """Verify sparse_bsr_tensor records a symbolic Sparse_bsr_tensor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sparse_bsr_tensor(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sparse_bsr_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_compressed_tensor() -> None:
    """Verify sparse_compressed_tensor records a symbolic Sparse_compressed_tensor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sparse_compressed_tensor(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sparse_compressed_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_coo_tensor() -> None:
    """Verify sparse_coo_tensor records a symbolic Sparse_coo_tensor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sparse_coo_tensor(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sparse_coo_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_csc_tensor() -> None:
    """Verify sparse_csc_tensor records a symbolic Sparse_csc_tensor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sparse_csc_tensor(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sparse_csc_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_sparse_csr_tensor() -> None:
    """Verify sparse_csr_tensor records a symbolic Sparse_csr_tensor tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sparse_csr_tensor(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sparse_csr_tensor_out"
    assert out.dtype == DType.FLOAT32


def test_split_copy() -> None:
    """Verify split_copy records a symbolic Split_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.split_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Split_copy_out"
    assert out.dtype == DType.FLOAT32


def test_split_with_sizes() -> None:
    """Verify split_with_sizes records a symbolic Split_with_sizes tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.split_with_sizes(x)
    assert isinstance(out, Tensor)
    assert out.name == "Split_with_sizes_out"
    assert out.dtype == DType.FLOAT32


def test_split_with_sizes_copy() -> None:
    """Verify split_with_sizes_copy records a symbolic Split_with_sizes_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.split_with_sizes_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Split_with_sizes_copy_out"
    assert out.dtype == DType.FLOAT32


def test_spmm() -> None:
    """Verify spmm records a symbolic Spmm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.spmm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Spmm_out"
    assert out.dtype == DType.FLOAT32


def test_sqrt() -> None:
    """Verify sqrt records a symbolic Sqrt tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sqrt(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sqrt_out"
    assert out.dtype == DType.FLOAT32


def test_sqrt_() -> None:
    """Verify sqrt_ records a symbolic Sqrt_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sqrt_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sqrt__out"
    assert out.dtype == DType.FLOAT32


def test_square() -> None:
    """Verify square records a symbolic Square tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.square(x)
    assert isinstance(out, Tensor)
    assert out.name == "Square_out"
    assert out.dtype == DType.FLOAT32


def test_square_() -> None:
    """Verify square_ records a symbolic Square_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.square_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Square__out"
    assert out.dtype == DType.FLOAT32


def test_squeeze() -> None:
    """Verify squeeze records a symbolic Squeeze tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.squeeze(x)
    assert isinstance(out, Tensor)
    assert out.name == "Squeeze_out"
    assert out.dtype == DType.FLOAT32


def test_squeeze_copy() -> None:
    """Verify squeeze_copy records a symbolic Squeeze_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.squeeze_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Squeeze_copy_out"
    assert out.dtype == DType.FLOAT32


def test_sspaddmm() -> None:
    """Verify sspaddmm records a symbolic Sspaddmm tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.sspaddmm(x)
    assert isinstance(out, Tensor)
    assert out.name == "Sspaddmm_out"
    assert out.dtype == DType.FLOAT32


def test_std() -> None:
    """Verify std records a symbolic Std tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.std(x)
    assert isinstance(out, Tensor)
    assert out.name == "Std_out"
    assert out.dtype == DType.FLOAT32


def test_std_mean() -> None:
    """Verify std_mean records a symbolic Std_mean tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.std_mean(x)
    assert isinstance(out, Tensor)
    assert out.name == "Std_mean_out"
    assert out.dtype == DType.FLOAT32


def test_stft() -> None:
    """Verify stft records a symbolic Stft tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.stft(x)
    assert isinstance(out, Tensor)
    assert out.name == "Stft_out"
    assert out.dtype == DType.FLOAT32


def test_subtract() -> None:
    """Verify subtract records a symbolic Subtract tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.subtract(x)
    assert isinstance(out, Tensor)
    assert out.name == "Subtract_out"
    assert out.dtype == DType.FLOAT32


def test_svd() -> None:
    """Verify svd records a symbolic Svd tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.svd(x)
    assert isinstance(out, Tensor)
    assert out.name == "Svd_out"
    assert out.dtype == DType.FLOAT32


def test_swapaxes() -> None:
    """Verify swapaxes records a symbolic Swapaxes tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.swapaxes(x)
    assert isinstance(out, Tensor)
    assert out.name == "Swapaxes_out"
    assert out.dtype == DType.FLOAT32


def test_swapdims() -> None:
    """Verify swapdims records a symbolic Swapdims tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.swapdims(x)
    assert isinstance(out, Tensor)
    assert out.name == "Swapdims_out"
    assert out.dtype == DType.FLOAT32


def test_t() -> None:
    """Verify t records a symbolic T tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.t(x)
    assert isinstance(out, Tensor)
    assert out.name == "T_out"
    assert out.dtype == DType.FLOAT32


def test_t_copy() -> None:
    """Verify t_copy records a symbolic T_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.t_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "T_copy_out"
    assert out.dtype == DType.FLOAT32


def test_take() -> None:
    """Verify take records a symbolic Take tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.take(x)
    assert isinstance(out, Tensor)
    assert out.name == "Take_out"
    assert out.dtype == DType.FLOAT32


def test_take_along_dim() -> None:
    """Verify take_along_dim records a symbolic Take_along_dim tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.take_along_dim(x)
    assert isinstance(out, Tensor)
    assert out.name == "Take_along_dim_out"
    assert out.dtype == DType.FLOAT32


def test_tan_() -> None:
    """Verify tan_ records a symbolic Tan_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.tan_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Tan__out"
    assert out.dtype == DType.FLOAT32


def test_tanh_() -> None:
    """Verify tanh_ records a symbolic Tanh_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.tanh_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Tanh__out"
    assert out.dtype == DType.FLOAT32


def test_tensor_split() -> None:
    """Verify tensor_split records a symbolic Tensor_split tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.tensor_split(x)
    assert isinstance(out, Tensor)
    assert out.name == "Tensor_split_out"
    assert out.dtype == DType.FLOAT32


def test_tensordot() -> None:
    """Verify tensordot records a symbolic Tensordot tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.tensordot(x)
    assert isinstance(out, Tensor)
    assert out.name == "Tensordot_out"
    assert out.dtype == DType.FLOAT32


def test_threshold() -> None:
    """Verify threshold records a symbolic Threshold tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.threshold(x)
    assert isinstance(out, Tensor)
    assert out.name == "Threshold_out"
    assert out.dtype == DType.FLOAT32


def test_threshold_() -> None:
    """Verify threshold_ records a symbolic Threshold_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.threshold_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Threshold__out"
    assert out.dtype == DType.FLOAT32


def test_transpose_copy() -> None:
    """Verify transpose_copy records a symbolic Transpose_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.transpose_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Transpose_copy_out"
    assert out.dtype == DType.FLOAT32


def test_trapezoid() -> None:
    """Verify trapezoid records a symbolic Trapezoid tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.trapezoid(x)
    assert isinstance(out, Tensor)
    assert out.name == "Trapezoid_out"
    assert out.dtype == DType.FLOAT32


def test_trapz() -> None:
    """Verify trapz records a symbolic Trapz tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.trapz(x)
    assert isinstance(out, Tensor)
    assert out.name == "Trapz_out"
    assert out.dtype == DType.FLOAT32


def test_triangular_solve() -> None:
    """Verify triangular_solve records a symbolic Triangular_solve tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.triangular_solve(x)
    assert isinstance(out, Tensor)
    assert out.name == "Triangular_solve_out"
    assert out.dtype == DType.FLOAT32


def test_tril() -> None:
    """Verify tril records a symbolic Tril tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.tril(x)
    assert isinstance(out, Tensor)
    assert out.name == "Tril_out"
    assert out.dtype == DType.FLOAT32


def test_tril_indices() -> None:
    """Verify tril_indices records a symbolic Tril_indices tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.tril_indices(x)
    assert isinstance(out, Tensor)
    assert out.name == "Tril_indices_out"
    assert out.dtype == DType.FLOAT32


def test_triplet_margin_loss() -> None:
    """Verify triplet_margin_loss records a symbolic Triplet_margin_loss tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.triplet_margin_loss(x)
    assert isinstance(out, Tensor)
    assert out.name == "Triplet_margin_loss_out"
    assert out.dtype == DType.FLOAT32


def test_triu() -> None:
    """Verify triu records a symbolic Triu tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.triu(x)
    assert isinstance(out, Tensor)
    assert out.name == "Triu_out"
    assert out.dtype == DType.FLOAT32


def test_triu_indices() -> None:
    """Verify triu_indices records a symbolic Triu_indices tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.triu_indices(x)
    assert isinstance(out, Tensor)
    assert out.name == "Triu_indices_out"
    assert out.dtype == DType.FLOAT32


def test_true_divide() -> None:
    """Verify true_divide records a symbolic True_divide tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.true_divide(x)
    assert isinstance(out, Tensor)
    assert out.name == "True_divide_out"
    assert out.dtype == DType.FLOAT32


def test_trunc() -> None:
    """Verify trunc records a symbolic Trunc tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.trunc(x)
    assert isinstance(out, Tensor)
    assert out.name == "Trunc_out"
    assert out.dtype == DType.FLOAT32


def test_trunc_() -> None:
    """Verify trunc_ records a symbolic Trunc_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.trunc_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Trunc__out"
    assert out.dtype == DType.FLOAT32


def test_unbind() -> None:
    """Verify unbind records a symbolic Unbind tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unbind(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unbind_out"
    assert out.dtype == DType.FLOAT32


def test_unbind_copy() -> None:
    """Verify unbind_copy records a symbolic Unbind_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unbind_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unbind_copy_out"
    assert out.dtype == DType.FLOAT32


def test_unflatten() -> None:
    """Verify unflatten records a symbolic Unflatten tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unflatten(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unflatten_out"
    assert out.dtype == DType.FLOAT32


def test_unfold_copy() -> None:
    """Verify unfold_copy records a symbolic Unfold_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unfold_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unfold_copy_out"
    assert out.dtype == DType.FLOAT32


def test_unique_consecutive() -> None:
    """Verify unique_consecutive records a symbolic Unique_consecutive tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unique_consecutive(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unique_consecutive_out"
    assert out.dtype == DType.FLOAT32


def test_unsafe_chunk() -> None:
    """Verify unsafe_chunk records a symbolic Unsafe_chunk tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unsafe_chunk(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unsafe_chunk_out"
    assert out.dtype == DType.FLOAT32


def test_unsafe_split() -> None:
    """Verify unsafe_split records a symbolic Unsafe_split tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unsafe_split(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unsafe_split_out"
    assert out.dtype == DType.FLOAT32


def test_unsafe_split_with_sizes() -> None:
    """Verify unsafe_split_with_sizes records a symbolic Unsafe_split_with_sizes tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unsafe_split_with_sizes(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unsafe_split_with_sizes_out"
    assert out.dtype == DType.FLOAT32


def test_unsqueeze() -> None:
    """Verify unsqueeze records a symbolic Unsqueeze tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unsqueeze(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unsqueeze_out"
    assert out.dtype == DType.FLOAT32


def test_unsqueeze_copy() -> None:
    """Verify unsqueeze_copy records a symbolic Unsqueeze_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.unsqueeze_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Unsqueeze_copy_out"
    assert out.dtype == DType.FLOAT32


def test_values_copy() -> None:
    """Verify values_copy records a symbolic Values_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.values_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Values_copy_out"
    assert out.dtype == DType.FLOAT32


def test_vander() -> None:
    """Verify vander records a symbolic Vander tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.vander(x)
    assert isinstance(out, Tensor)
    assert out.name == "Vander_out"
    assert out.dtype == DType.FLOAT32


def test_var() -> None:
    """Verify var records a symbolic Var tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.var(x)
    assert isinstance(out, Tensor)
    assert out.name == "Var_out"
    assert out.dtype == DType.FLOAT32


def test_var_mean() -> None:
    """Verify var_mean records a symbolic Var_mean tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.var_mean(x)
    assert isinstance(out, Tensor)
    assert out.name == "Var_mean_out"
    assert out.dtype == DType.FLOAT32


def test_vdot() -> None:
    """Verify vdot records a symbolic Vdot tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.vdot(x)
    assert isinstance(out, Tensor)
    assert out.name == "Vdot_out"
    assert out.dtype == DType.FLOAT32


def test_view_as_complex() -> None:
    """Verify view_as_complex records a symbolic View_as_complex tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.view_as_complex(x)
    assert isinstance(out, Tensor)
    assert out.name == "View_as_complex_out"
    assert out.dtype == DType.FLOAT32


def test_view_as_complex_copy() -> None:
    """Verify view_as_complex_copy records a symbolic View_as_complex_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.view_as_complex_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "View_as_complex_copy_out"
    assert out.dtype == DType.FLOAT32


def test_view_as_real() -> None:
    """Verify view_as_real records a symbolic View_as_real tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.view_as_real(x)
    assert isinstance(out, Tensor)
    assert out.name == "View_as_real_out"
    assert out.dtype == DType.FLOAT32


def test_view_as_real_copy() -> None:
    """Verify view_as_real_copy records a symbolic View_as_real_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.view_as_real_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "View_as_real_copy_out"
    assert out.dtype == DType.FLOAT32


def test_view_copy() -> None:
    """Verify view_copy records a symbolic View_copy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.view_copy(x)
    assert isinstance(out, Tensor)
    assert out.name == "View_copy_out"
    assert out.dtype == DType.FLOAT32


def test_vsplit() -> None:
    """Verify vsplit records a symbolic Vsplit tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.vsplit(x)
    assert isinstance(out, Tensor)
    assert out.name == "Vsplit_out"
    assert out.dtype == DType.FLOAT32


def test_vstack() -> None:
    """Verify vstack records a symbolic Vstack tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.vstack(x)
    assert isinstance(out, Tensor)
    assert out.name == "Vstack_out"
    assert out.dtype == DType.FLOAT32


def test_xlogy() -> None:
    """Verify xlogy records a symbolic Xlogy tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.xlogy(x)
    assert isinstance(out, Tensor)
    assert out.name == "Xlogy_out"
    assert out.dtype == DType.FLOAT32


def test_xlogy_() -> None:
    """Verify xlogy_ records a symbolic Xlogy_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.xlogy_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Xlogy__out"
    assert out.dtype == DType.FLOAT32


def test_zero_() -> None:
    """Verify zero_ records a symbolic Zero_ tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.zero_(x)
    assert isinstance(out, Tensor)
    assert out.name == "Zero__out"
    assert out.dtype == DType.FLOAT32


def test_zeros_like() -> None:
    """Verify zeros_like records a symbolic Zeros_like tensor."""
    x = Tensor(name="x", shape=(2, 2), dtype=DType.FLOAT32)
    out = torch_auto.zeros_like(x)
    assert isinstance(out, Tensor)
    assert out.name == "Zeros_like_out"
    assert out.dtype == DType.FLOAT32
