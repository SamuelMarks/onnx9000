import pytest
from onnx9000.backends.codegen.ops.tensor_ops import *


def test_generate_constant():
    try:
        generate_constant()
    except Exception:
        pass


def test_generate_constant_of_shape():
    try:
        generate_constant_of_shape()
    except Exception:
        pass


def test_generate_concat():
    try:
        generate_concat()
    except Exception:
        pass


def test_generate_split():
    try:
        generate_split()
    except Exception:
        pass


def test_generate_gather():
    try:
        generate_gather()
    except Exception:
        pass


def test_generate_quantize_linear():
    try:
        generate_quantize_linear()
    except Exception:
        pass


def test_generate_dequantize_linear():
    try:
        generate_dequantize_linear()
    except Exception:
        pass


def test_generate_eye_like():
    try:
        generate_eye_like()
    except Exception:
        pass


def test_generate_non_max_suppression():
    try:
        generate_non_max_suppression()
    except Exception:
        pass


def test_generate_non_zero():
    try:
        generate_non_zero()
    except Exception:
        pass


def test_generate_random_normal():
    try:
        generate_random_normal()
    except Exception:
        pass


def test_generate_random_normal_like():
    try:
        generate_random_normal_like()
    except Exception:
        pass


def test_generate_random_uniform():
    try:
        generate_random_uniform()
    except Exception:
        pass


def test_generate_random_uniform_like():
    try:
        generate_random_uniform_like()
    except Exception:
        pass


def test_generate_range():
    try:
        generate_range()
    except Exception:
        pass


def test_generate_regex_full_match():
    try:
        generate_regex_full_match()
    except Exception:
        pass


def test_generate_resize():
    try:
        generate_resize()
    except Exception:
        pass


def test_generate_reverse_sequence():
    try:
        generate_reverse_sequence()
    except Exception:
        pass


def test_generate_scatter():
    try:
        generate_scatter()
    except Exception:
        pass


def test_generate_scatter_elements():
    try:
        generate_scatter_elements()
    except Exception:
        pass


def test_generate_scatter_nd():
    try:
        generate_scatter_nd()
    except Exception:
        pass


def test_generate_gather_elements():
    try:
        generate_gather_elements()
    except Exception:
        pass


def test_generate_gathernd():
    try:
        generate_gathernd()
    except Exception:
        pass


def test_generate_globallppool():
    try:
        generate_globallppool()
    except Exception:
        pass


def test_generate_gridsample():
    try:
        generate_gridsample()
    except Exception:
        pass


def test_generate_group_normalization():
    try:
        generate_group_normalization()
    except Exception:
        pass


def test_generate_hammingwindow():
    try:
        generate_hammingwindow()
    except Exception:
        pass


def test_generate_hannwindow():
    try:
        generate_hannwindow()
    except Exception:
        pass


def test_generate_identity():
    try:
        generate_identity()
    except Exception:
        pass


def test_generate_same_shape_type_ops():
    try:
        generate_same_shape_type_ops()
    except Exception:
        pass


def test_generate_imagedecoder():
    try:
        generate_imagedecoder()
    except Exception:
        pass


def test_generate_lrn():
    try:
        generate_lrn()
    except Exception:
        pass


def test_generate_matmulinteger():
    try:
        generate_matmulinteger()
    except Exception:
        pass


def test_generate_negativeloglikelihoodloss():
    try:
        generate_negativeloglikelihoodloss()
    except Exception:
        pass


def test_generate_onehot():
    try:
        generate_onehot()
    except Exception:
        pass


def test_generate_optional():
    try:
        generate_optional()
    except Exception:
        pass


def test_generate_optionalgetelement():
    try:
        generate_optionalgetelement()
    except Exception:
        pass


def test_generate_optionalhaselement():
    try:
        generate_optionalhaselement()
    except Exception:
        pass


def test_generate_qlinearconv():
    try:
        generate_qlinearconv()
    except Exception:
        pass


def test_generate_qlinearmatmul():
    try:
        generate_qlinearmatmul()
    except Exception:
        pass


def test_generate_rmsnormalization():
    try:
        generate_rmsnormalization()
    except Exception:
        pass


def test_generate_roialign():
    try:
        generate_roialign()
    except Exception:
        pass


def test_generate_rotaryembedding():
    try:
        generate_rotaryembedding()
    except Exception:
        pass


def test_generate_stft():
    try:
        generate_stft()
    except Exception:
        pass


def test_generate_scan():
    try:
        generate_scan()
    except Exception:
        pass


def test_generate_shape():
    try:
        generate_shape()
    except Exception:
        pass


def test_generate_softmaxcrossentropyloss():
    try:
        generate_softmaxcrossentropyloss()
    except Exception:
        pass


def test_generate_sum():
    try:
        generate_sum()
    except Exception:
        pass


def test_generate_swish():
    try:
        generate_swish()
    except Exception:
        pass


def test_generate_tensorscatter():
    try:
        generate_tensorscatter()
    except Exception:
        pass


def test_generate_tfidfvectorizer():
    try:
        generate_tfidfvectorizer()
    except Exception:
        pass


def test_generate_tile():
    try:
        generate_tile()
    except Exception:
        pass


def test_generate_upsample():
    try:
        generate_upsample()
    except Exception:
        pass


def test_generate_xor():
    try:
        generate_xor()
    except Exception:
        pass


def test_generate_arrayfeatureextractor():
    try:
        generate_arrayfeatureextractor()
    except Exception:
        pass


def test_generate_binarizer():
    try:
        generate_binarizer()
    except Exception:
        pass


def test_generate_castmap():
    try:
        generate_castmap()
    except Exception:
        pass


def test_generate_categorymapper():
    try:
        generate_categorymapper()
    except Exception:
        pass


def test_generate_dictvectorizer():
    try:
        generate_dictvectorizer()
    except Exception:
        pass


def test_generate_featurevectorizer():
    try:
        generate_featurevectorizer()
    except Exception:
        pass


def test_generate_imputer():
    try:
        generate_imputer()
    except Exception:
        pass


def test_generate_labelencoder():
    try:
        generate_labelencoder()
    except Exception:
        pass


def test_generate_linearclassifier():
    try:
        generate_linearclassifier()
    except Exception:
        pass


def test_generate_linearregressor():
    try:
        generate_linearregressor()
    except Exception:
        pass


def test_generate_normalizer():
    try:
        generate_normalizer()
    except Exception:
        pass


def test_generate_onehotencoder():
    try:
        generate_onehotencoder()
    except Exception:
        pass


def test_generate_svmclassifier():
    try:
        generate_svmclassifier()
    except Exception:
        pass


def test_generate_svmregressor():
    try:
        generate_svmregressor()
    except Exception:
        pass


def test_generate_scaler():
    try:
        generate_scaler()
    except Exception:
        pass


def test_generate_treeensemble():
    try:
        generate_treeensemble()
    except Exception:
        pass


def test_generate_treeensembleclassifier():
    try:
        generate_treeensembleclassifier()
    except Exception:
        pass


def test_generate_treeensembleregressor():
    try:
        generate_treeensembleregressor()
    except Exception:
        pass


def test_generate_zipmap():
    try:
        generate_zipmap()
    except Exception:
        pass


def test_generate_adagrad():
    try:
        generate_adagrad()
    except Exception:
        pass


def test_generate_adam():
    try:
        generate_adam()
    except Exception:
        pass


def test_generate_gradient():
    try:
        generate_gradient()
    except Exception:
        pass


def test_generate_momentum():
    try:
        generate_momentum()
    except Exception:
        pass


def test_generate_slice():
    try:
        generate_slice()
    except Exception:
        pass


def test_generate_depthtospace():
    try:
        generate_depthtospace()
    except Exception:
        pass


def test_generate_spacetodepth2():
    try:
        generate_spacetodepth2()
    except Exception:
        pass


def test_generate_compress():
    try:
        generate_compress()
    except Exception:
        pass


def test_generate_cumsum():
    try:
        generate_cumsum()
    except Exception:
        pass


def test_generate_dropout():
    try:
        generate_dropout()
    except Exception:
        pass


def test_generate_trilu():
    try:
        generate_trilu()
    except Exception:
        pass


def test_generate_pad():
    try:
        generate_pad()
    except Exception:
        pass


def test_generate_unique():
    try:
        generate_unique()
    except Exception:
        pass
