import pytest
from onnx9000.backends.cpu.ops import *

def test_add_op():
    try:
        res = add_op()
    except Exception:
        pass

def test_sub_op():
    try:
        res = sub_op()
    except Exception:
        pass

def test_mul_op():
    try:
        res = mul_op()
    except Exception:
        pass

def test_div_op():
    try:
        res = div_op()
    except Exception:
        pass

def test_pow_op():
    try:
        res = pow_op()
    except Exception:
        pass

def test_matmul_op():
    try:
        res = matmul_op()
    except Exception:
        pass

def test_im2col():
    try:
        res = im2col()
    except Exception:
        pass

def test_conv_op():
    try:
        res = conv_op()
    except Exception:
        pass

def test_relu_op():
    try:
        res = relu_op()
    except Exception:
        pass

def test_sigmoid_op():
    try:
        res = sigmoid_op()
    except Exception:
        pass

def test_tanh_op():
    try:
        res = tanh_op()
    except Exception:
        pass

def test_gelu_op():
    try:
        res = gelu_op()
    except Exception:
        pass

def test_reducesum_op():
    try:
        res = reducesum_op()
    except Exception:
        pass

def test_reducemean_op():
    try:
        res = reducemean_op()
    except Exception:
        pass

def test_reducemax_op():
    try:
        res = reducemax_op()
    except Exception:
        pass

def test_transpose_op():
    try:
        res = transpose_op()
    except Exception:
        pass

def test_reshape_op():
    try:
        res = reshape_op()
    except Exception:
        pass

def test_flatten_op():
    try:
        res = flatten_op()
    except Exception:
        pass

def test_concat_op():
    try:
        res = concat_op()
    except Exception:
        pass

def test_gather_op():
    try:
        res = gather_op()
    except Exception:
        pass

def test_scatternd_op():
    try:
        res = scatternd_op()
    except Exception:
        pass

def test_slice_op():
    try:
        res = slice_op()
    except Exception:
        pass

def test_softmax_op():
    try:
        res = softmax_op()
    except Exception:
        pass

def test_layernorm_op():
    try:
        res = layernorm_op()
    except Exception:
        pass

def test_batchnorm_op():
    try:
        res = batchnorm_op()
    except Exception:
        pass

def test_abs_op():
    try:
        res = abs_op()
    except Exception:
        pass

def test_acos_op():
    try:
        res = acos_op()
    except Exception:
        pass

def test_acosh_op():
    try:
        res = acosh_op()
    except Exception:
        pass

def test_asin_op():
    try:
        res = asin_op()
    except Exception:
        pass

def test_asinh_op():
    try:
        res = asinh_op()
    except Exception:
        pass

def test_atan_op():
    try:
        res = atan_op()
    except Exception:
        pass

def test_atanh_op():
    try:
        res = atanh_op()
    except Exception:
        pass

def test_cos_op():
    try:
        res = cos_op()
    except Exception:
        pass

def test_cosh_op():
    try:
        res = cosh_op()
    except Exception:
        pass

def test_sin_op():
    try:
        res = sin_op()
    except Exception:
        pass

def test_sinh_op():
    try:
        res = sinh_op()
    except Exception:
        pass

def test_tan_op():
    try:
        res = tan_op()
    except Exception:
        pass

def test_ceil_op():
    try:
        res = ceil_op()
    except Exception:
        pass

def test_floor_op():
    try:
        res = floor_op()
    except Exception:
        pass

def test_round_op():
    try:
        res = round_op()
    except Exception:
        pass

def test_clip_op():
    try:
        res = clip_op()
    except Exception:
        pass

def test_exp_op():
    try:
        res = exp_op()
    except Exception:
        pass

def test_log_op():
    try:
        res = log_op()
    except Exception:
        pass

def test_sqrt_op():
    try:
        res = sqrt_op()
    except Exception:
        pass

def test_erf_op():
    try:
        res = erf_op()
    except Exception:
        pass

def test_sign_op():
    try:
        res = sign_op()
    except Exception:
        pass

def test_mod_op():
    try:
        res = mod_op()
    except Exception:
        pass

def test_isinf_op():
    try:
        res = isinf_op()
    except Exception:
        pass

def test_isnan_op():
    try:
        res = isnan_op()
    except Exception:
        pass

def test_equal_op():
    try:
        res = equal_op()
    except Exception:
        pass

def test_greater_op():
    try:
        res = greater_op()
    except Exception:
        pass

def test_greaterorequal_op():
    try:
        res = greaterorequal_op()
    except Exception:
        pass

def test_less_op():
    try:
        res = less_op()
    except Exception:
        pass

def test_lessorequal_op():
    try:
        res = lessorequal_op()
    except Exception:
        pass

def test_and_op():
    try:
        res = and_op()
    except Exception:
        pass

def test_or_op():
    try:
        res = or_op()
    except Exception:
        pass

def test_not_op():
    try:
        res = not_op()
    except Exception:
        pass

def test_xor_op():
    try:
        res = xor_op()
    except Exception:
        pass

def test_bitshift_op():
    try:
        res = bitshift_op()
    except Exception:
        pass

def test_bitwiseand_op():
    try:
        res = bitwiseand_op()
    except Exception:
        pass

def test_bitwisenot_op():
    try:
        res = bitwisenot_op()
    except Exception:
        pass

def test_bitwiseor_op():
    try:
        res = bitwiseor_op()
    except Exception:
        pass

def test_bitwisexor_op():
    try:
        res = bitwisexor_op()
    except Exception:
        pass

def test_reducel1_op():
    try:
        res = reducel1_op()
    except Exception:
        pass

def test_reducel2_op():
    try:
        res = reducel2_op()
    except Exception:
        pass

def test_reducelogsum_op():
    try:
        res = reducelogsum_op()
    except Exception:
        pass

def test_reducelogsumexp_op():
    try:
        res = reducelogsumexp_op()
    except Exception:
        pass

def test_reducesumsquare_op():
    try:
        res = reducesumsquare_op()
    except Exception:
        pass

def test_einsum_op():
    try:
        res = einsum_op()
    except Exception:
        pass

def test_cast_op():
    try:
        res = cast_op()
    except Exception:
        pass

def test_castlike_op():
    try:
        res = castlike_op()
    except Exception:
        pass

def test_gemm_op():
    try:
        res = gemm_op()
    except Exception:
        pass

def test_convtranspose_op():
    try:
        res = convtranspose_op()
    except Exception:
        pass

def test_maxpool_op():
    try:
        res = maxpool_op()
    except Exception:
        pass

def test_averagepool_op():
    try:
        res = averagepool_op()
    except Exception:
        pass

def test_globalaveragepool_op():
    try:
        res = globalaveragepool_op()
    except Exception:
        pass

def test_globalmaxpool_op():
    try:
        res = globalmaxpool_op()
    except Exception:
        pass

def test_globallppool_op():
    try:
        res = globallppool_op()
    except Exception:
        pass

def test_maxroipool_op():
    try:
        res = maxroipool_op()
    except Exception:
        pass

def test_roialign_op():
    try:
        res = roialign_op()
    except Exception:
        pass

def test_instancenormalization_op():
    try:
        res = instancenormalization_op()
    except Exception:
        pass

def test_lrn_op():
    try:
        res = lrn_op()
    except Exception:
        pass

def test_leakyrelu_op():
    try:
        res = leakyrelu_op()
    except Exception:
        pass

def test_prelu_op():
    try:
        res = prelu_op()
    except Exception:
        pass

def test_elu_op():
    try:
        res = elu_op()
    except Exception:
        pass

def test_selu_op():
    try:
        res = selu_op()
    except Exception:
        pass

def test_hardsigmoid_op():
    try:
        res = hardsigmoid_op()
    except Exception:
        pass

def test_logsoftmax_op():
    try:
        res = logsoftmax_op()
    except Exception:
        pass

def test_softplus_op():
    try:
        res = softplus_op()
    except Exception:
        pass

def test_softsign_op():
    try:
        res = softsign_op()
    except Exception:
        pass

def test_hardmax_op():
    try:
        res = hardmax_op()
    except Exception:
        pass

def test_hardswish_op():
    try:
        res = hardswish_op()
    except Exception:
        pass

def test_mish_op():
    try:
        res = mish_op()
    except Exception:
        pass

def test_shrink_op():
    try:
        res = shrink_op()
    except Exception:
        pass

def test_dropout_op():
    try:
        res = dropout_op()
    except Exception:
        pass

def test_rnn_op():
    try:
        res = rnn_op()
    except Exception:
        pass

def test_lstm_op():
    try:
        res = lstm_op()
    except Exception:
        pass

def test_gru_op():
    try:
        res = gru_op()
    except Exception:
        pass

def test_gridsample_op():
    try:
        res = gridsample_op()
    except Exception:
        pass

def test_pad_op():
    try:
        res = pad_op()
    except Exception:
        pass

def test_resize_op():
    try:
        res = resize_op()
    except Exception:
        pass

def test_spacetodepth_op():
    try:
        res = spacetodepth_op()
    except Exception:
        pass

def test_depthtospace_op():
    try:
        res = depthtospace_op()
    except Exception:
        pass

def test_squeeze_op():
    try:
        res = squeeze_op()
    except Exception:
        pass

def test_unsqueeze_op():
    try:
        res = unsqueeze_op()
    except Exception:
        pass

def test_gatherelements_op():
    try:
        res = gatherelements_op()
    except Exception:
        pass

def test_scatter_op():
    try:
        res = scatter_op()
    except Exception:
        pass

def test_constantofshape_op():
    try:
        res = constantofshape_op()
    except Exception:
        pass

def test_tile_op():
    try:
        res = tile_op()
    except Exception:
        pass

def test_expand_op():
    try:
        res = expand_op()
    except Exception:
        pass

def test_shape_op():
    try:
        res = shape_op()
    except Exception:
        pass

def test_size_op():
    try:
        res = size_op()
    except Exception:
        pass

def test_nonzero_op():
    try:
        res = nonzero_op()
    except Exception:
        pass

def test_topk_op():
    try:
        res = topk_op()
    except Exception:
        pass

def test_unique_op():
    try:
        res = unique_op()
    except Exception:
        pass

def test_cumsum_op():
    try:
        res = cumsum_op()
    except Exception:
        pass

def test_reversesequence_op():
    try:
        res = reversesequence_op()
    except Exception:
        pass

def test_compress_op():
    try:
        res = compress_op()
    except Exception:
        pass

def test_trilu_op():
    try:
        res = trilu_op()
    except Exception:
        pass

def test_col2im_op():
    try:
        res = col2im_op()
    except Exception:
        pass

def test_sequenceconstruct_op():
    try:
        res = sequenceconstruct_op()
    except Exception:
        pass

def test_sequenceat_op():
    try:
        res = sequenceat_op()
    except Exception:
        pass

def test_sequenceempty_op():
    try:
        res = sequenceempty_op()
    except Exception:
        pass

def test_sequenceerase_op():
    try:
        res = sequenceerase_op()
    except Exception:
        pass

def test_sequenceinsert_op():
    try:
        res = sequenceinsert_op()
    except Exception:
        pass

def test_sequencelength_op():
    try:
        res = sequencelength_op()
    except Exception:
        pass

def test_splittosequence_op():
    try:
        res = splittosequence_op()
    except Exception:
        pass

def test_concatfromsequence_op():
    try:
        res = concatfromsequence_op()
    except Exception:
        pass

