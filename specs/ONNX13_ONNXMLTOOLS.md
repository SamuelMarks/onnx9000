# onnxmltools Replication & Parity Tracker

## Description
This document tracks the complete reimplementation of `onnxmltools` within the `onnx9000` ecosystem.
The original `onnxmltools` relies heavily on native bindings for LightGBM, XGBoost, CatBoost, CoreML, and PySpark to perform conversions. 
Our `onnx9000` reimplementation uses pure-Python parsers (reading raw JSON dumps, protobufs, or binary structures) to translate these diverse model types into `ai.onnx.ml` and `ai.onnx` topologies. 
This zero-dependency approach means you can convert a 10,000-tree XGBoost model or a complex SparkML pipeline entirely in a web browser using WASM or in a cold-start AWS Lambda function without installing multi-gigabyte ML frameworks.

## Exhaustive Parity Checklist

### 1. Pure-Python LightGBM Conversion (40+ items)
- [ ] Implement zero-dependency parser for LightGBM `.txt` dump format
- [ ] Implement zero-dependency parser for LightGBM JSON dump format
- [ ] Parse LightGBM `tree_info` (left/right children, thresholds, features)
- [ ] Extract LightGBM leaf values and weights accurately
- [ ] Map LightGBM `regression` objective -> `TreeEnsembleRegressor`
- [ ] Map LightGBM `regression_l1` objective -> `TreeEnsembleRegressor`
- [ ] Map LightGBM `huber` objective -> `TreeEnsembleRegressor`
- [ ] Map LightGBM `fair` objective -> `TreeEnsembleRegressor`
- [ ] Map LightGBM `poisson` objective -> `TreeEnsembleRegressor` + `Exp`
- [ ] Map LightGBM `quantile` objective -> `TreeEnsembleRegressor`
- [ ] Map LightGBM `mape` objective -> `TreeEnsembleRegressor`
- [ ] Map LightGBM `gamma` objective -> `TreeEnsembleRegressor` + `Exp`
- [ ] Map LightGBM `tweedie` objective -> `TreeEnsembleRegressor` + `Exp`
- [ ] Map LightGBM `binary` objective -> `TreeEnsembleClassifier`
- [ ] Map LightGBM `multiclass` objective -> `TreeEnsembleClassifier`
- [ ] Map LightGBM `multiclassova` objective -> `TreeEnsembleClassifier`
- [ ] Map LightGBM `cross_entropy` objective -> `TreeEnsembleClassifier`
- [ ] Map LightGBM `cross_entropy_lambda` objective -> `TreeEnsembleClassifier`
- [ ] Map LightGBM `lambdarank` objective -> `TreeEnsembleRegressor`
- [ ] Map LightGBM `rank_xendcg` objective -> `TreeEnsembleRegressor`
- [ ] Map LightGBM missing value logic (default to left/right) to `nodes_missing_value_tracks_true`
- [ ] Extract and map LightGBM `class_names` to ONNX `classlabels_strings`
- [ ] Extract and map LightGBM `class_names` to ONNX `classlabels_ints`
- [ ] Support LightGBM models with categorical feature splits natively in ONNX
- [ ] Map LightGBM `base_score` / `objective_seed` to ONNX base values
- [ ] Map LightGBM `sigmoid` parameter for binary classification
- [ ] Map LightGBM `num_class` correctly for multiclass ZipMap generation
- [ ] Verify LightGBM `boosting_type="gbdt"` translation
- [ ] Verify LightGBM `boosting_type="dart"` translation (handling dropout scaling)
- [ ] Verify LightGBM `boosting_type="goss"` translation
- [ ] Verify LightGBM `boosting_type="rf"` translation (Random Forest mode)

### 2. Pure-Python XGBoost Conversion (40+ items)
- [ ] Implement zero-dependency parser for XGBoost JSON model dumps
- [ ] Implement zero-dependency parser for XGBoost legacy binary `.ubj` (Universal Binary JSON)
- [ ] Parse XGBoost `trees` array (nodes, splits, Yes/No/Missing paths)
- [ ] Extract XGBoost `base_score` and apply dynamically based on objective
- [ ] Map XGBoost `reg:squarederror` -> `TreeEnsembleRegressor`
- [ ] Map XGBoost `reg:squaredlogerror` -> `TreeEnsembleRegressor`
- [ ] Map XGBoost `reg:logistic` -> `TreeEnsembleRegressor` + `Sigmoid`
- [ ] Map XGBoost `reg:pseudohubererror` -> `TreeEnsembleRegressor`
- [ ] Map XGBoost `binary:logistic` -> `TreeEnsembleClassifier` (with Sigmoid post-transform)
- [ ] Map XGBoost `binary:logitraw` -> `TreeEnsembleClassifier` (with NONE post-transform)
- [ ] Map XGBoost `binary:hinge` -> `TreeEnsembleClassifier` (with Step post-transform logic)
- [ ] Map XGBoost `count:poisson` -> `TreeEnsembleRegressor` + `Exp`
- [ ] Map XGBoost `survival:cox` -> `TreeEnsembleRegressor` + `Exp`
- [ ] Map XGBoost `survival:aft` -> `TreeEnsembleRegressor`
- [ ] Map XGBoost `multi:softmax` -> `TreeEnsembleClassifier` (with Softmax post-transform)
- [ ] Map XGBoost `multi:softprob` -> `TreeEnsembleClassifier` (with Softmax probability generation)
- [ ] Map XGBoost `rank:pairwise` -> `TreeEnsembleRegressor`
- [ ] Map XGBoost `rank:ndcg` -> `TreeEnsembleRegressor`
- [ ] Map XGBoost `rank:map` -> `TreeEnsembleRegressor`
- [ ] Map XGBoost missing value logic (`missing` attribute) to ONNX tree node rules
- [ ] Track XGBoost `tree_limit` / `best_iteration` for truncated model conversion
- [ ] Map XGBoost multi-output regression natively (Forest regressor with multiple target ids)
- [ ] Reconstruct categorical features based on XGBoost experimental categorical split nodes
- [ ] Support XGBoost `gblinear` booster -> `LinearRegressor` / `LinearClassifier`
- [ ] Support XGBoost `dart` booster -> `TreeEnsemble` (applying weight drops statically)
- [ ] Support XGBoost `gbtree` booster natively
- [ ] Parse XGBoost `feature_names` to establish explicit ONNX inputs
- [ ] Map XGBoost `scale_pos_weight` behavior implicitly within tree leaves
- [ ] Support Scikit-Learn wrapper `XGBClassifier`
- [ ] Support Scikit-Learn wrapper `XGBRegressor`

### 3. Pure-Python CatBoost Conversion (30+ items)
- [ ] Implement zero-dependency parser for CatBoost JSON dumps
- [ ] Parse CatBoost oblivious trees (flattening symmetric tree structures to standard DAGs)
- [ ] Extract CatBoost float features and splits
- [ ] Extract CatBoost one-hot encoded categorical features
- [ ] Extract CatBoost CTR (Categorical Target Encoding) features natively if possible
- [ ] Map CatBoost `RMSE` / `MultiRMSE` -> `TreeEnsembleRegressor`
- [ ] Map CatBoost `MAE` / `Quantile` -> `TreeEnsembleRegressor`
- [ ] Map CatBoost `Logloss` -> `TreeEnsembleClassifier`
- [ ] Map CatBoost `CrossEntropy` -> `TreeEnsembleClassifier`
- [ ] Map CatBoost `MultiClass` -> `TreeEnsembleClassifier` (Softmax)
- [ ] Map CatBoost `MultiClassOneVsAll` -> `TreeEnsembleClassifier` (Sigmoid)
- [ ] Map CatBoost `Poisson` -> `TreeEnsembleRegressor`
- [ ] Map CatBoost `SurvivalAft` -> `TreeEnsembleRegressor`
- [ ] Extract CatBoost leaf values accurately from oblivious tree arrays
- [ ] Extract CatBoost `scale_and_bias` attributes and map to ONNX Affine transform / Tree biases
- [ ] Translate symmetric thresholds to `ai.onnx.ml` `nodes_values` correctly
- [ ] Unroll oblivious tree bitmasks into standard ONNX left/right index structures
- [ ] Support `CatBoostClassifier` wrappers
- [ ] Support `CatBoostRegressor` wrappers

### 4. CoreML Protobuf Conversion - Core & ML (40+ items)
- [ ] Implement zero-dependency parser for CoreML `.mlmodel` (Protobuf)
- [ ] Parse CoreML `ModelDescription` for inputs, outputs, and feature types
- [ ] Map CoreML `Int64FeatureType` -> ONNX `Int64`
- [ ] Map CoreML `DoubleFeatureType` -> ONNX `Float64` or `Float32`
- [ ] Map CoreML `StringFeatureType` -> ONNX `String`
- [ ] Map CoreML `ImageFeatureType` -> ONNX `Tensor` (Image shapes)
- [ ] Map CoreML `MultiArrayFeatureType` -> ONNX `Tensor`
- [ ] Map CoreML `DictionaryFeatureType` -> ONNX `Map` / `Sequence`
- [ ] Map CoreML `TreeEnsembleClassifier` -> `ai.onnx.ml.TreeEnsembleClassifier`
- [ ] Map CoreML `TreeEnsembleRegressor` -> `ai.onnx.ml.TreeEnsembleRegressor`
- [ ] Map CoreML `SupportVectorClassifier` -> `ai.onnx.ml.SVMClassifier`
- [ ] Map CoreML `SupportVectorRegressor` -> `ai.onnx.ml.SVMRegressor`
- [ ] Map CoreML `GLMClassifier` -> `ai.onnx.ml.LinearClassifier`
- [ ] Map CoreML `GLMRegressor` -> `ai.onnx.ml.LinearRegressor`
- [ ] Map CoreML `DictVectorizer` -> `ai.onnx.ml.DictVectorizer`
- [ ] Map CoreML `FeatureVectorizer` -> `ai.onnx.ml.FeatureVectorizer`
- [ ] Map CoreML `Imputer` -> `ai.onnx.ml.Imputer`
- [ ] Map CoreML `Scaler` -> `ai.onnx.ml.Scaler`
- [ ] Map CoreML `Normalizer` -> `ai.onnx.ml.Normalizer`
- [ ] Map CoreML `OneHotEncoder` -> `ai.onnx.ml.OneHotEncoder`
- [ ] Map CoreML `CategoricalMapping` -> `ai.onnx.ml.CategoryMapper`
- [ ] Map CoreML `ArrayFeatureExtractor` -> `ai.onnx.ml.ArrayFeatureExtractor`
- [ ] Map CoreML `NonMaximumSuppression` -> `ai.onnx.NonMaxSuppression`
- [ ] Map CoreML `ItemSimilarityRecommender` -> ONNX Custom Subgraph
- [ ] Map CoreML `WordTagger` -> ONNX Custom Subgraph
- [ ] Map CoreML `TextClassifier` -> ONNX Custom Subgraph
- [ ] Map CoreML `VisionFeaturePrint` -> ONNX Extensibility
- [ ] Map CoreML Pipeline models (recursive conversion)

### 5. CoreML Protobuf Conversion - Neural Networks (40+ items)
- [ ] Map CoreML `NeuralNetwork` -> ONNX Subgraph
- [ ] Map CoreML `NeuralNetworkClassifier` -> ONNX Subgraph + Probabilities
- [ ] Map CoreML `NeuralNetworkRegressor` -> ONNX Subgraph + Scores
- [ ] Map CoreML `ConvolutionLayer` -> `Conv`
- [ ] Map CoreML `PoolingLayer` -> `MaxPool` / `AveragePool`
- [ ] Map CoreML `ActivationReLU` -> `Relu`
- [ ] Map CoreML `ActivationLeakyReLU` -> `LeakyRelu`
- [ ] Map CoreML `ActivationSigmoid` -> `Sigmoid`
- [ ] Map CoreML `ActivationTanh` -> `Tanh`
- [ ] Map CoreML `ActivationLinear` -> `Add` + `Mul`
- [ ] Map CoreML `ActivationPReLU` -> `PRelu`
- [ ] Map CoreML `ActivationELU` -> `Elu`
- [ ] Map CoreML `ActivationSoftsign` -> `Softsign`
- [ ] Map CoreML `ActivationSoftplus` -> `Softplus`
- [ ] Map CoreML `ActivationParametricSoftmax` -> `Softmax`
- [ ] Map CoreML `BatchnormLayer` -> `BatchNormalization`
- [ ] Map CoreML `InnerProductLayer` -> `Gemm` / `MatMul`
- [ ] Map CoreML `SoftmaxLayer` -> `Softmax`
- [ ] Map CoreML `FlattenLayer` -> `Flatten`
- [ ] Map CoreML `ConcatLayer` -> `Concat`
- [ ] Map CoreML `ReshapeLayer` -> `Reshape`
- [ ] Map CoreML `PaddingLayer` -> `Pad`
- [ ] Map CoreML `PermuteLayer` -> `Transpose`
- [ ] Map CoreML `UpsampleLayer` -> `Upsample` / `Resize`
- [ ] Map CoreML `L2NormalizeLayer` -> `LpNormalization`
- [ ] Map CoreML `SimpleRNNLayer` -> `RNN`
- [ ] Map CoreML `GRULayer` -> `GRU`
- [ ] Map CoreML `UniDirectionalLSTMLayer` -> `LSTM`
- [ ] Map CoreML `BiDirectionalLSTMLayer` -> `LSTM`
- [ ] Map CoreML `ScaleLayer` -> `Mul`
- [ ] Map CoreML `CropLayer` -> `Slice`
- [ ] Map CoreML `AverageLayer` -> `Mean`
- [ ] Map CoreML `MaxLayer` -> `Max`
- [ ] Map CoreML `MinLayer` -> `Min`
- [ ] Map CoreML `DotProductLayer` -> `Mul` + `ReduceSum`
- [ ] Map CoreML `ReduceLayer` -> `Reduce*` operations

### 6. SparkML Pipeline & Transformer Conversion (40+ items)
- [ ] Implement zero-dependency parser for SparkML Pipeline JSON/Parquet dumps
- [ ] Extract PySpark `PipelineModel` stages
- [ ] Map Spark `Binarizer` -> `ai.onnx.ml.Binarizer`
- [ ] Map Spark `Bucketizer` -> `ai.onnx.ml.CategoryMapper` / Custom Subgraph
- [ ] Map Spark `ChiSqSelector` -> `ai.onnx.ml.ArrayFeatureExtractor`
- [ ] Map Spark `CountVectorizerModel` -> `ai.onnx.ml.CountVectorizer`
- [ ] Map Spark `DCT` -> ONNX Custom Subgraph
- [ ] Map Spark `ElementwiseProduct` -> `Mul`
- [ ] Map Spark `HashingTF` -> ONNX Custom (MurmurHash3)
- [ ] Map Spark `IDFModel` -> `Mul` (with extracted IDF weights)
- [ ] Map Spark `ImputerModel` -> `ai.onnx.ml.Imputer`
- [ ] Map Spark `IndexToString` -> `ai.onnx.ml.CategoryMapper`
- [ ] Map Spark `MaxAbsScalerModel` -> `ai.onnx.ml.Scaler`
- [ ] Map Spark `MinMaxScalerModel` -> `ai.onnx.ml.Scaler`
- [ ] Map Spark `NGram` -> ONNX Sequence logic
- [ ] Map Spark `Normalizer` -> `ai.onnx.ml.Normalizer`
- [ ] Map Spark `OneHotEncoderModel` -> `ai.onnx.ml.OneHotEncoder`
- [ ] Map Spark `PCAModel` -> `MatMul`
- [ ] Map Spark `PolynomialExpansion` -> Math Subgraph
- [ ] Map Spark `QuantileDiscretizer` -> `ai.onnx.ml.Binarizer` / Subgraph
- [ ] Map Spark `RegexTokenizer` -> ONNX Custom Regex Node
- [ ] Map Spark `StandardScalerModel` -> `ai.onnx.ml.Scaler`
- [ ] Map Spark `StopWordsRemover` -> ONNX Custom Subgraph
- [ ] Map Spark `StringIndexerModel` -> `ai.onnx.ml.CategoryMapper`
- [ ] Map Spark `Tokenizer` -> ONNX StringSplit / Custom
- [ ] Map Spark `VectorAssembler` -> `ai.onnx.ml.FeatureVectorizer` / `Concat`
- [ ] Map Spark `VectorIndexerModel` -> Subgraph
- [ ] Map Spark `VectorSlicer` -> `ai.onnx.ml.ArrayFeatureExtractor` / `Slice`
- [ ] Map Spark `Word2VecModel` -> `Gather` (Embedding lookup)

### 7. SparkML Classifier & Regressor Conversion (30+ items)
- [ ] Map Spark `LogisticRegressionModel` -> `ai.onnx.ml.LinearClassifier`
- [ ] Map Spark `DecisionTreeClassificationModel` -> `ai.onnx.ml.TreeEnsembleClassifier`
- [ ] Map Spark `RandomForestClassificationModel` -> `ai.onnx.ml.TreeEnsembleClassifier`
- [ ] Map Spark `GBTClassificationModel` -> `ai.onnx.ml.TreeEnsembleClassifier`
- [ ] Map Spark `MultilayerPerceptronClassificationModel` -> Chained `MatMul` + `Sigmoid`
- [ ] Map Spark `LinearSVCModel` -> `ai.onnx.ml.SVMClassifier`
- [ ] Map Spark `NaiveBayesModel` -> Probability Subgraph
- [ ] Map Spark `LinearRegressionModel` -> `ai.onnx.ml.LinearRegressor`
- [ ] Map Spark `GeneralizedLinearRegressionModel` -> `ai.onnx.ml.LinearRegressor` + Link function
- [ ] Map Spark `DecisionTreeRegressionModel` -> `ai.onnx.ml.TreeEnsembleRegressor`
- [ ] Map Spark `RandomForestRegressionModel` -> `ai.onnx.ml.TreeEnsembleRegressor`
- [ ] Map Spark `GBTRegressionModel` -> `ai.onnx.ml.TreeEnsembleRegressor`
- [ ] Map Spark `AFTSurvivalRegressionModel` -> Regression Subgraph
- [ ] Map Spark `IsotonicRegressionModel` -> Subgraph / Custom
- [ ] Map Spark `FMClassificationModel` (Factorization Machines) -> Math Subgraph
- [ ] Map Spark `FMRegressionModel` -> Math Subgraph
- [ ] Map Spark `KMeansModel` -> Distance Subgraph + `ArgMin`
- [ ] Map Spark `BisectingKMeansModel` -> Distance Subgraph
- [ ] Map Spark `GaussianMixtureModel` -> Probability Subgraph

### 8. LibSVM & Misc Model Support (10+ items)
- [ ] Implement zero-dependency parser for LibSVM model text formats
- [ ] Map LibSVM `C-SVC` -> `ai.onnx.ml.SVMClassifier`
- [ ] Map LibSVM `nu-SVC` -> `ai.onnx.ml.SVMClassifier`
- [ ] Map LibSVM `one-class SVM` -> `ai.onnx.ml.SVMClassifier`
- [ ] Map LibSVM `epsilon-SVR` -> `ai.onnx.ml.SVMRegressor`
- [ ] Map LibSVM `nu-SVR` -> `ai.onnx.ml.SVMRegressor`
- [ ] Extract SVM support vectors, dual coefficients, and rhos
- [ ] Map linear, polynomial, RBF, and sigmoid kernels cleanly
- [ ] Support probability estimates (Platt scaling) via LibSVM parameters
- [ ] Integrate H2O model parsing (MOJO/POJO to ONNX logic mapped if structurally requested)

### 9. Graph Optimizations & Routing (20+ items)
- [ ] Implement TreeEnsemble node compression (removing redundant leaf nodes)
- [ ] Implement VectorAssembler fusion (canceling out consecutive un-assemblies)
- [ ] Optimize sequential Spark StringIndexers into a single dict lookup
- [ ] Resolve LightGBM/XGBoost `int64` vs `float32` split thresholds globally
- [ ] Handle TreeEnsemble batch inference dimension alignment explicitly
- [ ] Flatten nested `ZipMap` operators from multiple estimators into a single dictionary array
- [ ] Remove `Cast` operations that do not change precision semantics
- [ ] Fuse scaling nodes generated by CoreML pipelines
- [ ] Apply constant folding on CoreML NeuralNetwork normalization weights
- [ ] Enforce deterministic node naming strategies across all converters

### 10. Lightweight Runtime & Zero-Dependency Capabilities (20+ items)
- [ ] Execute LightGBM parsing fully in memory without disk I/O
- [ ] Execute XGBoost parsing fully in memory without disk I/O
- [ ] No `lightgbm` pip package required at runtime
- [ ] No `xgboost` pip package required at runtime
- [ ] No `catboost` pip package required at runtime
- [ ] No `coremltools` or MacOS required at runtime (parse `.mlmodel` anywhere)
- [ ] No `pyspark` or Java JVM required at runtime
- [ ] Provide TypeScript/JS native wrappers for `onnx9000` web translation
- [ ] Emscripten build configuration for translating models inside Safari/Chrome
- [ ] Cloudflare Worker ready: script fits within 1MB worker limits
- [ ] AWS Lambda ready: instantaneous cold start translation of tree structures
- [ ] Support handling models > 2GB using file-backed memory-mapping structures
- [ ] Support dynamic ONNX shape overriding natively in all 5 sub-converters
- [ ] Produce strict `ai.onnx.ml` Opsets 1, 2, 3, 4 based on user flags
- [ ] Provide 100% strict compliance validation against standard `onnxruntime`

### 11. Advanced CatBoost Loss Functions & Edge Cases (30+ items)
- [ ] Map CatBoost `Huber` -> `TreeEnsembleRegressor`
- [ ] Map CatBoost `Lq` -> `TreeEnsembleRegressor`
- [ ] Map CatBoost `Tweedie` -> `TreeEnsembleRegressor` + `Exp`
- [ ] Map CatBoost `Focal` -> `TreeEnsembleClassifier`
- [ ] Map CatBoost `BrierScore` -> `TreeEnsembleClassifier`
- [ ] Map CatBoost `HingeLoss` -> `TreeEnsembleClassifier`
- [ ] Map CatBoost `HammingLoss` -> `TreeEnsembleClassifier`
- [ ] Map CatBoost `ZeroOneLoss` -> `TreeEnsembleClassifier`
- [ ] Map CatBoost `PairLogit` -> `TreeEnsembleClassifier` (pairwise)
- [ ] Map CatBoost `PairLogitPairwise` -> `TreeEnsembleClassifier`
- [ ] Map CatBoost `YetiRank` -> `TreeEnsembleRegressor`
- [ ] Map CatBoost `YetiRankPairwise` -> `TreeEnsembleRegressor`
- [ ] Map CatBoost `QueryRMSE` -> `TreeEnsembleRegressor`
- [ ] Map CatBoost `QuerySoftMax` -> `TreeEnsembleClassifier`
- [ ] Map CatBoost `StochasticFilter` -> `TreeEnsembleClassifier`
- [ ] Map CatBoost `CrossEntropy` (with smoothed weights) -> `TreeEnsembleClassifier`
- [ ] Support CatBoost `text` features (via ONNX string hashing/TF-IDF)
- [ ] Support CatBoost `embedding` features (via ONNX Gather)
- [ ] Flatten nested oblivious trees correctly when `leaf_values` exceed standard depths
- [ ] Handle CatBoost explicit `border_count` constraints internally

### 12. Advanced SparkML & Pyspark Ecosystem Ops (20+ items)
- [ ] Handle Spark `SparseVector` objects intrinsically via ONNX dense/sparse structures
- [ ] Map Spark `SQLTransformer` -> ONNX Custom logic/Subgraph (where mathematically possible)
- [ ] Map Spark `Binarizer` with custom `threshold` vectors
- [ ] Map Spark `Bucketizer` with `splitsArray`
- [ ] Map Spark `StringIndexer` with `handleInvalid='skip'`
- [ ] Map Spark `StringIndexer` with `handleInvalid='keep'`
- [ ] Map Spark `OneHotEncoder` with `dropLast=True`
- [ ] Map Spark `OneHotEncoder` with `dropLast=False`
- [ ] Map Spark `VectorAssembler` with `handleInvalid='skip'`
- [ ] Map Spark `VectorAssembler` with `handleInvalid='keep'`
- [ ] Map Spark `MinMaxScaler` with custom `min`/`max` bounds
- [ ] Map Spark `MaxAbsScaler` with zero-variance protections
- [ ] Support PySpark 2.x `PipelineModel` definitions
- [ ] Support PySpark 3.x `PipelineModel` definitions

### 13. Advanced H2O & MOJO Translation (20+ items)
- [ ] Implement zero-dependency parser for H2O MOJO / POJO structures
- [ ] Map H2O `DistributedRandomForest` -> `TreeEnsembleClassifier`
- [ ] Map H2O `GradientBoostingEstimator` -> `TreeEnsembleRegressor` / Classifier
- [ ] Map H2O `DeepLearningEstimator` -> ONNX Subgraph (MLP)
- [ ] Map H2O `GeneralizedLinearEstimator` -> `LinearRegressor` / Classifier
- [ ] Map H2O `IsolationForest` -> `TreeEnsembleClassifier` (Anomaly detection)
- [ ] Map H2O `KMeans` -> Distance Subgraph
- [ ] Map H2O `PCA` -> `MatMul`
- [ ] Map H2O `NaiveBayes` -> Probability Subgraph
- [ ] Extract H2O categorical encodings seamlessly into ONNX `CategoryMapper`
- [ ] Handle H2O missing values natively via MOJO node properties

### 14. Testing, Web Parity, and Validation (15+ items)
- [ ] Unit test: End-to-end `lightgbm` Regressor (browser environment)
- [ ] Unit test: End-to-end `lightgbm` Classifier (WASM environment)
- [ ] Unit test: End-to-end `xgboost` Regressor (browser environment)
- [ ] Unit test: End-to-end `xgboost` Classifier (WASM environment)
- [ ] Unit test: End-to-end `catboost` Regressor (browser environment)
- [ ] Unit test: End-to-end `catboost` Classifier (WASM environment)
- [ ] Unit test: End-to-end `coreml` Image Classifier (WebGPU environment)
- [ ] Unit test: End-to-end `pyspark` Pipeline (WASM environment)
- [ ] Stress Test: 20,000 node XGBoost tree translated inside Chrome < 50ms
- [ ] Validate deterministic identical translations (pure Python vs native converters)
- [ ] Validate `ai.onnx.ml` execution against `onnx9000` Python JIT

### 15. LightGBM, XGBoost & LibSVM Hyper-edge Cases (30+ items)
- [ ] Parse LightGBM `custom` objective functions (if mathematically representable in ONNX)
- [ ] Map LightGBM `pos_bagging_fraction` impacts correctly
- [ ] Map LightGBM `neg_bagging_fraction` impacts correctly
- [ ] Handle LightGBM `categorical_feature` indexing natively in ONNX (no pre-processing needed)
- [ ] Support LightGBM `is_unbalance=True` leaf value adjustments
- [ ] Support LightGBM `scale_pos_weight` explicit overrides
- [ ] Map XGBoost `survival:aft` with specific exponential constraints
- [ ] Map XGBoost `survival:cox` hazard ratios to ONNX outputs
- [ ] Map XGBoost `binary:logitraw` -> `LinearClassifier` or Tree with no link function
- [ ] Handle XGBoost `base_margin` dynamically as an explicit ONNX graph input
- [ ] Parse XGBoost `interaction_constraints` strictly
- [ ] Map XGBoost `monotone_constraints` into tree evaluations correctly
- [ ] Support LibSVM `probability=1` -> `ai.onnx.ml.SVMClassifier` (with Platt Scaling parameters)
- [ ] Support LibSVM `probability=0` -> `ai.onnx.ml.SVMClassifier` (raw scores)
- [ ] Map LibSVM `shrinking=1` heuristics into SVM structures
- [ ] Ensure LibSVM `cache_size` is ignored (not applicable at inference)
- [ ] Map LibSVM `kernel_type=0` (linear) -> SVM
- [ ] Map LibSVM `kernel_type=1` (polynomial) -> SVM
- [ ] Map LibSVM `kernel_type=2` (RBF) -> SVM
- [ ] Map LibSVM `kernel_type=3` (sigmoid) -> SVM
- [ ] Extract LibSVM `degree` exactly
- [ ] Extract LibSVM `gamma` exactly
- [ ] Extract LibSVM `coef0` exactly
- [ ] Parse custom string tokens safely within `LibSVM` format lines
