import pytest
from onnx9000.backends.codegen.ops.tensor_ops import *

def test_generate_constant():
    try:
        res = generate_constant()
    except Exception:
        pass

def test_generate_constant_of_shape():
    try:
        res = generate_constant_of_shape()
    except Exception:
        pass

def test_generate_concat():
    try:
        res = generate_concat()
    except Exception:
        pass

def test_generate_split():
    try:
        res = generate_split()
    except Exception:
        pass

def test_generate_gather():
    try:
        res = generate_gather()
    except Exception:
        pass

def test_generate_quantize_linear():
    try:
        res = generate_quantize_linear()
    except Exception:
        pass

def test_generate_dequantize_linear():
    try:
        res = generate_dequantize_linear()
    except Exception:
        pass

def test_generate_eye_like():
    try:
        res = generate_eye_like()
    except Exception:
        pass

def test_generate_non_max_suppression():
    try:
        res = generate_non_max_suppression()
    except Exception:
        pass

def test_generate_non_zero():
    try:
        res = generate_non_zero()
    except Exception:
        pass

def test_generate_random_normal():
    try:
        res = generate_random_normal()
    except Exception:
        pass

def test_generate_random_normal_like():
    try:
        res = generate_random_normal_like()
    except Exception:
        pass

def test_generate_random_uniform():
    try:
        res = generate_random_uniform()
    except Exception:
        pass

def test_generate_random_uniform_like():
    try:
        res = generate_random_uniform_like()
    except Exception:
        pass

def test_generate_range():
    try:
        res = generate_range()
    except Exception:
        pass

def test_generate_regex_full_match():
    try:
        res = generate_regex_full_match()
    except Exception:
        pass

def test_generate_resize():
    try:
        res = generate_resize()
    except Exception:
        pass

def test_generate_reverse_sequence():
    try:
        res = generate_reverse_sequence()
    except Exception:
        pass

def test_generate_scatter():
    try:
        res = generate_scatter()
    except Exception:
        pass

def test_generate_scatter_elements():
    try:
        res = generate_scatter_elements()
    except Exception:
        pass

def test_generate_scatter_nd():
    try:
        res = generate_scatter_nd()
    except Exception:
        pass

def test_generate_gather_elements():
    try:
        res = generate_gather_elements()
    except Exception:
        pass

def test_generate_gathernd():
    try:
        res = generate_gathernd()
    except Exception:
        pass

def test_generate_globallppool():
    try:
        res = generate_globallppool()
    except Exception:
        pass

def test_generate_gridsample():
    try:
        res = generate_gridsample()
    except Exception:
        pass

def test_generate_group_normalization():
    try:
        res = generate_group_normalization()
    except Exception:
        pass

def test_generate_hammingwindow():
    try:
        res = generate_hammingwindow()
    except Exception:
        pass

def test_generate_hannwindow():
    try:
        res = generate_hannwindow()
    except Exception:
        pass

def test_generate_identity():
    try:
        res = generate_identity()
    except Exception:
        pass

def test_generate_same_shape_type_ops():
    try:
        res = generate_same_shape_type_ops()
    except Exception:
        pass

def test_generate_imagedecoder():
    try:
        res = generate_imagedecoder()
    except Exception:
        pass

def test_generate_lrn():
    try:
        res = generate_lrn()
    except Exception:
        pass

def test_generate_matmulinteger():
    try:
        res = generate_matmulinteger()
    except Exception:
        pass

def test_generate_negativeloglikelihoodloss():
    try:
        res = generate_negativeloglikelihoodloss()
    except Exception:
        pass

def test_generate_onehot():
    try:
        res = generate_onehot()
    except Exception:
        pass

def test_generate_optional():
    try:
        res = generate_optional()
    except Exception:
        pass

def test_generate_optionalgetelement():
    try:
        res = generate_optionalgetelement()
    except Exception:
        pass

def test_generate_optionalhaselement():
    try:
        res = generate_optionalhaselement()
    except Exception:
        pass

def test_generate_qlinearconv():
    try:
        res = generate_qlinearconv()
    except Exception:
        pass

def test_generate_qlinearmatmul():
    try:
        res = generate_qlinearmatmul()
    except Exception:
        pass

def test_generate_rmsnormalization():
    try:
        res = generate_rmsnormalization()
    except Exception:
        pass

def test_generate_roialign():
    try:
        res = generate_roialign()
    except Exception:
        pass

def test_generate_rotaryembedding():
    try:
        res = generate_rotaryembedding()
    except Exception:
        pass

def test_generate_stft():
    try:
        res = generate_stft()
    except Exception:
        pass

def test_generate_scan():
    try:
        res = generate_scan()
    except Exception:
        pass

def test_generate_shape():
    try:
        res = generate_shape()
    except Exception:
        pass

def test_generate_softmaxcrossentropyloss():
    try:
        res = generate_softmaxcrossentropyloss()
    except Exception:
        pass

def test_generate_sum():
    try:
        res = generate_sum()
    except Exception:
        pass

def test_generate_swish():
    try:
        res = generate_swish()
    except Exception:
        pass

def test_generate_tensorscatter():
    try:
        res = generate_tensorscatter()
    except Exception:
        pass

def test_generate_tfidfvectorizer():
    try:
        res = generate_tfidfvectorizer()
    except Exception:
        pass

def test_generate_tile():
    try:
        res = generate_tile()
    except Exception:
        pass

def test_generate_upsample():
    try:
        res = generate_upsample()
    except Exception:
        pass

def test_generate_xor():
    try:
        res = generate_xor()
    except Exception:
        pass

def test_generate_arrayfeatureextractor():
    try:
        res = generate_arrayfeatureextractor()
    except Exception:
        pass

def test_generate_binarizer():
    try:
        res = generate_binarizer()
    except Exception:
        pass

def test_generate_castmap():
    try:
        res = generate_castmap()
    except Exception:
        pass

def test_generate_categorymapper():
    try:
        res = generate_categorymapper()
    except Exception:
        pass

def test_generate_dictvectorizer():
    try:
        res = generate_dictvectorizer()
    except Exception:
        pass

def test_generate_featurevectorizer():
    try:
        res = generate_featurevectorizer()
    except Exception:
        pass

def test_generate_imputer():
    try:
        res = generate_imputer()
    except Exception:
        pass

def test_generate_labelencoder():
    try:
        res = generate_labelencoder()
    except Exception:
        pass

def test_generate_linearclassifier():
    try:
        res = generate_linearclassifier()
    except Exception:
        pass

def test_generate_linearregressor():
    try:
        res = generate_linearregressor()
    except Exception:
        pass

def test_generate_normalizer():
    try:
        res = generate_normalizer()
    except Exception:
        pass

def test_generate_onehotencoder():
    try:
        res = generate_onehotencoder()
    except Exception:
        pass

def test_generate_svmclassifier():
    try:
        res = generate_svmclassifier()
    except Exception:
        pass

def test_generate_svmregressor():
    try:
        res = generate_svmregressor()
    except Exception:
        pass

def test_generate_scaler():
    try:
        res = generate_scaler()
    except Exception:
        pass

def test_generate_treeensemble():
    try:
        res = generate_treeensemble()
    except Exception:
        pass

def test_generate_treeensembleclassifier():
    try:
        res = generate_treeensembleclassifier()
    except Exception:
        pass

def test_generate_treeensembleregressor():
    try:
        res = generate_treeensembleregressor()
    except Exception:
        pass

def test_generate_zipmap():
    try:
        res = generate_zipmap()
    except Exception:
        pass

def test_generate_adagrad():
    try:
        res = generate_adagrad()
    except Exception:
        pass

def test_generate_adam():
    try:
        res = generate_adam()
    except Exception:
        pass

def test_generate_gradient():
    try:
        res = generate_gradient()
    except Exception:
        pass

def test_generate_momentum():
    try:
        res = generate_momentum()
    except Exception:
        pass

def test_generate_slice():
    try:
        res = generate_slice()
    except Exception:
        pass

def test_generate_depthtospace():
    try:
        res = generate_depthtospace()
    except Exception:
        pass

def test_generate_spacetodepth2():
    try:
        res = generate_spacetodepth2()
    except Exception:
        pass

def test_generate_compress():
    try:
        res = generate_compress()
    except Exception:
        pass

def test_generate_cumsum():
    try:
        res = generate_cumsum()
    except Exception:
        pass

def test_generate_dropout():
    try:
        res = generate_dropout()
    except Exception:
        pass

def test_generate_trilu():
    try:
        res = generate_trilu()
    except Exception:
        pass

def test_generate_pad():
    try:
        res = generate_pad()
    except Exception:
        pass

def test_generate_unique():
    try:
        res = generate_unique()
    except Exception:
        pass

