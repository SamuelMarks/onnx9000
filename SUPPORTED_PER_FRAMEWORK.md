# Supported Frameworks Coverage

This file tracks the level of support for various ML frameworks in ONNX9000.

## Summary

| Target | Supported | Total | Percentage |
|---|---|---|---|
| ONNX Spec | 160 | 200 | 80.00% |
| Torch | 25 | 914 | 2.74% |
| Tensorflow | 177 | 316 | 56.01% |
| Keras | 47 | Unknown | N/A |
| Jax | 1 | 134 | 0.75% |
| Flax | 1 | 11 | 9.09% |
| Paddle | 96 | Unknown | N/A |
| Coremltools | 0 | 27 | 0.00% |
| Sklearn | 115 | 43 | 100.00% |
| Xgboost | 2 | Unknown | N/A |
| Lightgbm | 2 | Unknown | N/A |
| Catboost | 2 | 24 | 8.33% |
| Pyspark | 1 | 22 | 4.55% |
| H2o | 1 | Unknown | N/A |
| Libsvm | 1 | Unknown | N/A |
| Cntk | 0 | Unknown | N/A |
| Mxnet | 0 | Unknown | N/A |
| Caffe | 0 | Unknown | N/A |
| Gguf | 2 | 207 | 0.97% |
| Safetensors | 2 | 5 | 40.00% |

## ONNX Spec Coverage

**Coverage:** 160/200 (80.00%)

**Commit:** [`ff324324bb7ef0c8508b1aa402271febbfe39e4e`](https://github.com/onnx/onnx/commit/ff324324bb7ef0c8508b1aa402271febbfe39e4e)


## Framework Versions

| Framework | Version |
|---|---|
| onnx | 1.20.1 |
| torch | 2.11.0 |
| tensorflow | 2.21.0 |
| keras | Not Installed |
| jax | 0.9.2 |
| flax | 0.12.6 |
| paddle | Not Installed |
| coremltools | 9.0 |
| sklearn | 1.8.0 |
| xgboost | Not Installed |
| lightgbm | Not Installed |
| catboost | 1.2.10 |
| pyspark | 4.1.1 |
| h2o | Not Installed |
| libsvm | Not Installed |
| cntk | Not Installed |
| mxnet | Not Installed |
| caffe | Not Installed |
| gguf | unknown |
| safetensors | 0.7.0 |

## Detailed Operators

| ONNX Operator | ONNX9000 |
|---|---|
| Abs | ✅ |
| Acos | ✅ |
| Acosh | ✅ |
| Add | ✅ |
| AffineGrid | ✅ |
| And | ✅ |
| ArgMax | ✅ |
| ArgMin | ✅ |
| Asin | ✅ |
| Asinh | ✅ |
| Atan | ✅ |
| Atanh | ✅ |
| Attention | ✅ |
| AveragePool | ✅ |
| BatchNormalization | ❌ |
| Bernoulli | ✅ |
| BitCast | ❌ |
| BitShift | ✅ |
| BitwiseAnd | ✅ |
| BitwiseNot | ✅ |
| BitwiseOr | ✅ |
| BitwiseXor | ✅ |
| BlackmanWindow | ✅ |
| Cast | ✅ |
| CastLike | ✅ |
| Ceil | ✅ |
| Celu | ✅ |
| CenterCropPad | ✅ |
| Clip | ✅ |
| Col2Im | ✅ |
| Compress | ✅ |
| Concat | ✅ |
| ConcatFromSequence | ✅ |
| Constant | ✅ |
| ConstantOfShape | ✅ |
| Conv | ✅ |
| ConvInteger | ✅ |
| ConvTranspose | ✅ |
| Cos | ✅ |
| Cosh | ✅ |
| CumProd | ❌ |
| CumSum | ✅ |
| DFT | ✅ |
| DeformConv | ✅ |
| DepthToSpace | ✅ |
| DequantizeLinear | ✅ |
| Det | ✅ |
| Div | ✅ |
| Dropout | ✅ |
| DynamicQuantizeLinear | ✅ |
| Einsum | ✅ |
| Elu | ✅ |
| Equal | ✅ |
| Erf | ✅ |
| Exp | ✅ |
| Expand | ✅ |
| EyeLike | ✅ |
| Flatten | ❌ |
| Floor | ✅ |
| GRU | ✅ |
| Gather | ✅ |
| GatherElements | ✅ |
| GatherND | ✅ |
| Gelu | ❌ |
| Gemm | ✅ |
| GlobalAveragePool | ✅ |
| GlobalLpPool | ❌ |
| GlobalMaxPool | ✅ |
| Greater | ✅ |
| GreaterOrEqual | ✅ |
| GridSample | ❌ |
| GroupNormalization | ✅ |
| HammingWindow | ❌ |
| HannWindow | ❌ |
| HardSigmoid | ✅ |
| HardSwish | ✅ |
| Hardmax | ✅ |
| Identity | ❌ |
| If | ❌ |
| ImageDecoder | ❌ |
| InstanceNormalization | ✅ |
| IsInf | ✅ |
| IsNaN | ✅ |
| LRN | ✅ |
| LSTM | ✅ |
| LayerNormalization | ✅ |
| LeakyRelu | ✅ |
| Less | ✅ |
| LessOrEqual | ✅ |
| Log | ❌ |
| LogSoftmax | ✅ |
| Loop | ❌ |
| LpNormalization | ✅ |
| LpPool | ✅ |
| MatMul | ✅ |
| MatMulInteger | ❌ |
| Max | ✅ |
| MaxPool | ✅ |
| MaxRoiPool | ❌ |
| MaxUnpool | ❌ |
| Mean | ✅ |
| MeanVarianceNormalization | ✅ |
| MelWeightMatrix | ✅ |
| Min | ✅ |
| Mish | ✅ |
| Mod | ✅ |
| Mul | ✅ |
| Multinomial | ✅ |
| Neg | ✅ |
| NegativeLogLikelihoodLoss | ❌ |
| NonMaxSuppression | ✅ |
| NonZero | ✅ |
| Not | ✅ |
| OneHot | ✅ |
| Optional | ❌ |
| OptionalGetElement | ❌ |
| OptionalHasElement | ❌ |
| Or | ✅ |
| PRelu | ✅ |
| Pad | ❌ |
| Pow | ✅ |
| QLinearConv | ❌ |
| QLinearMatMul | ❌ |
| QuantizeLinear | ✅ |
| RMSNormalization | ❌ |
| RNN | ✅ |
| RandomNormal | ✅ |
| RandomNormalLike | ✅ |
| RandomUniform | ✅ |
| RandomUniformLike | ✅ |
| Range | ❌ |
| Reciprocal | ✅ |
| ReduceL1 | ✅ |
| ReduceL2 | ✅ |
| ReduceLogSum | ✅ |
| ReduceLogSumExp | ✅ |
| ReduceMax | ✅ |
| ReduceMean | ✅ |
| ReduceMin | ✅ |
| ReduceProd | ✅ |
| ReduceSum | ✅ |
| ReduceSumSquare | ✅ |
| RegexFullMatch | ✅ |
| Relu | ✅ |
| Reshape | ✅ |
| Resize | ✅ |
| ReverseSequence | ✅ |
| RoiAlign | ❌ |
| RotaryEmbedding | ❌ |
| Round | ✅ |
| STFT | ❌ |
| Scan | ❌ |
| Scatter | ✅ |
| ScatterElements | ✅ |
| ScatterND | ✅ |
| Selu | ✅ |
| SequenceAt | ✅ |
| SequenceConstruct | ✅ |
| SequenceEmpty | ✅ |
| SequenceErase | ✅ |
| SequenceInsert | ✅ |
| SequenceLength | ✅ |
| SequenceMap | ✅ |
| Shape | ❌ |
| Shrink | ✅ |
| Sigmoid | ✅ |
| Sign | ✅ |
| Sin | ✅ |
| Sinh | ✅ |
| Size | ✅ |
| Slice | ✅ |
| Softmax | ✅ |
| SoftmaxCrossEntropyLoss | ❌ |
| Softplus | ✅ |
| Softsign | ✅ |
| SpaceToDepth | ✅ |
| Split | ❌ |
| SplitToSequence | ✅ |
| Sqrt | ❌ |
| Squeeze | ❌ |
| StringConcat | ✅ |
| StringNormalizer | ✅ |
| StringSplit | ✅ |
| Sub | ✅ |
| Sum | ✅ |
| Swish | ✅ |
| Tan | ✅ |
| Tanh | ✅ |
| TensorScatter | ❌ |
| TfIdfVectorizer | ❌ |
| ThresholdedRelu | ✅ |
| Tile | ✅ |
| TopK | ✅ |
| Transpose | ✅ |
| Trilu | ✅ |
| Unique | ✅ |
| Unsqueeze | ❌ |
| Upsample | ❌ |
| Where | ✅ |
| Xor | ❌ |