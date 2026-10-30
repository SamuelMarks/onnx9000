"""Tensor manipulation, shape, slicing, indexing, and string ONNX operators."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import AttributeValue, Tensor
from onnx9000.core.ops.base import record_op
from onnx9000.core.registry import register_op


@register_op("ai.onnx", "Cast")
def cast(x: Tensor, to_type: int) -> Tensor:
    """Compute Cast."""
    return record_op("Cast", [x], {"to": to_type})


@register_op("ai.onnx", "CastLike")
def cast_like(x: Tensor, target: Tensor) -> Tensor:
    """Compute CastLike."""
    return record_op("CastLike", [x, target])


@register_op("ai.onnx", "Constant")
def constant(value: AttributeValue, dtype: int = 1) -> Tensor:
    """Compute Constant."""
    return record_op("Constant", [], {"value": value, "dtype": dtype})


@register_op("ai.onnx", "ConstantOfShape")
def constant_of_shape(input: Tensor, value: AttributeValue = 0.0, dtype: int = 1) -> Tensor:
    """Compute ConstantOfShape."""
    return record_op("ConstantOfShape", [input], {"value": value, "dtype": dtype})


@register_op("ai.onnx", "Concat")
def concat(inputs: list[Tensor], axis: int) -> Tensor:
    """Compute Concat."""
    return record_op("Concat", inputs, {"axis": axis})


@register_op("ai.onnx", "EyeLike")
def eye_like(input: Tensor, dtype: int = 1, k: int = 0) -> Tensor:
    """Compute EyeLike."""
    return record_op("EyeLike", [input], {"dtype": dtype, "k": k})


@register_op("onnx9000.custom", "Roll")
def roll(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Roll for windowed attention."""
    return record_op("Roll", [x], kwargs)


@register_op("onnx9000.custom", "Unroll")
def unroll(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Unroll for windowed attention."""
    return record_op("Unroll", [x], kwargs)


@register_op("onnx9000.custom", "Tokenizer")
def tokenizer(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Tokenization."""
    return record_op("Tokenizer", [x], kwargs)


@register_op("onnx9000.custom", "TextVectorization")
def text_vectorization(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Text Vectorization."""
    return record_op("TextVectorization", [x], kwargs)


@register_op("ai.onnx", "Multinomial")
def multinomial(input: Tensor, dtype: int = 6, sample_size: int = 1, seed: float = 0.0) -> Tensor:
    """Compute Multinomial."""
    return record_op(
        "Multinomial",
        [input],
        {"dtype": dtype, "sample_size": sample_size, "seed": seed},
    )


@register_op("ai.onnx", "NonMaxSuppression")
def non_max_suppression(
    boxes: Tensor,
    scores: Tensor,
    max_output_boxes_per_class: Tensor = None,
    iou_threshold: Tensor = None,
    score_threshold: Tensor = None,
    center_point_box: int = 0,
) -> Tensor:
    """Compute NonMaxSuppression."""
    inputs = [boxes, scores]
    if max_output_boxes_per_class is not None:
        inputs.append(max_output_boxes_per_class)
    if iou_threshold is not None:
        if max_output_boxes_per_class is None:
            raise Onnx9000Error("ONNX sequential inputs rule violation for NonMaxSuppression")
        inputs.append(iou_threshold)
    if score_threshold is not None:
        if iou_threshold is None:
            raise Onnx9000Error("ONNX sequential inputs rule violation for NonMaxSuppression")
        inputs.append(score_threshold)
    return record_op("NonMaxSuppression", inputs, {"center_point_box": center_point_box})


@register_op("ai.onnx", "NonZero")
def non_zero(x: Tensor) -> Tensor:
    """Compute NonZero."""
    return record_op("NonZero", [x])


@register_op("ai.onnx", "RandomNormal")
def random_normal(
    dtype: int = 1,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float = 0.0,
    shape: list[int] | None = None,
) -> Tensor:
    """Compute RandomNormal."""
    if shape is None:
        shape = []
    return record_op(
        "RandomNormal",
        [],
        {"dtype": dtype, "mean": mean, "scale": scale, "seed": seed, "shape": shape},
    )


@register_op("ai.onnx", "RandomNormalLike")
def random_normal_like(
    input: Tensor,
    dtype: int | None = None,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float = 0.0,
) -> Tensor:
    """Compute RandomNormalLike."""
    attrs: dict[str, AttributeValue] = {"mean": mean, "scale": scale, "seed": seed}
    if dtype is not None:
        attrs["dtype"] = dtype
    return record_op("RandomNormalLike", [input], attrs)


@register_op("ai.onnx", "RandomUniform")
def random_uniform(
    dtype: int = 1,
    high: float = 1.0,
    low: float = 0.0,
    seed: float = 0.0,
    shape: list[int] | None = None,
) -> Tensor:
    """Compute RandomUniform."""
    if shape is None:
        shape = []
    return record_op(
        "RandomUniform",
        [],
        {"dtype": dtype, "high": high, "low": low, "seed": seed, "shape": shape},
    )


@register_op("ai.onnx", "RandomUniformLike")
def random_uniform_like(
    input: Tensor,
    dtype: int | None = None,
    high: float = 1.0,
    low: float = 0.0,
    seed: float = 0.0,
) -> Tensor:
    """Compute RandomUniformLike."""
    attrs: dict[str, AttributeValue] = {"high": high, "low": low, "seed": seed}
    if dtype is not None:
        attrs["dtype"] = dtype
    return record_op("RandomUniformLike", [input], attrs)


@register_op("ai.onnx", "Range")
def range_op(start: Tensor, limit: Tensor, delta: Tensor) -> Tensor:
    """Compute Range."""
    return record_op("Range", [start, limit, delta])


@register_op("ai.onnx", "RegexFullMatch")
def regex_full_match(x: Tensor, pattern: str) -> Tensor:
    """Compute RegexFullMatch."""
    return record_op("RegexFullMatch", [x], {"pattern": pattern})


@register_op("ai.onnx", "Resize")
def resize(
    x: Tensor,
    roi: Tensor = None,
    scales: Tensor = None,
    sizes: Tensor = None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Compute Resize."""
    inputs = [x]
    if roi is not None:
        inputs.append(roi)
    elif scales is not None or sizes is not None:
        inputs.append(record_op("Constant", [], {"value": []}))
    if scales is not None:
        inputs.append(scales)
    elif sizes is not None:
        inputs.append(record_op("Constant", [], {"value": []}))
    if sizes is not None:
        inputs.append(sizes)
    return record_op("Resize", inputs, kwargs)


@register_op("ai.onnx", "Scatter")
def scatter(data: Tensor, indices: Tensor, updates: Tensor, axis: int = 0) -> Tensor:
    """Compute Scatter."""
    return record_op("Scatter", [data, indices, updates], {"axis": axis})


@register_op("ai.onnx", "ScatterElements")
def scatter_elements(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    axis: int = 0,
    reduction: str = "none",
) -> Tensor:
    """Compute ScatterElements."""
    return record_op(
        "ScatterElements",
        [data, indices, updates],
        {"axis": axis, "reduction": reduction},
    )


@register_op("ai.onnx", "ScatterND")
def scatter_nd(data: Tensor, indices: Tensor, updates: Tensor, reduction: str = "none") -> Tensor:
    """Compute ScatterND."""
    return record_op("ScatterND", [data, indices, updates], {"reduction": reduction})


@register_op("ai.onnx", "Size")
def size(data: Tensor) -> Tensor:
    """Compute Size."""
    return record_op("Size", [data])


@register_op("ai.onnx", "StringConcat")
def string_concat(x: Tensor, y: Tensor) -> Tensor:
    """Compute StringConcat."""
    return record_op("StringConcat", [x, y])


@register_op("ai.onnx", "StringNormalizer")
def string_normalizer(
    x: Tensor,
    case_change_action: str = "NONE",
    is_case_sensitive: int = 0,
    locale: str = "",
    stopwords: list[str] | None = None,
) -> Tensor:
    """Compute StringNormalizer."""
    attrs: dict[str, AttributeValue] = {
        "case_change_action": case_change_action,
        "is_case_sensitive": is_case_sensitive,
        "locale": locale,
    }
    if stopwords is not None:
        attrs["stopwords"] = stopwords
    return record_op("StringNormalizer", [x], attrs)


@register_op("ai.onnx", "StringSplit")
def string_split(x: Tensor, delimiter: str = "", maxsplit: int = -1) -> Tensor:
    """Compute StringSplit."""
    return record_op("StringSplit", [x], {"delimiter": delimiter, "maxsplit": maxsplit})


@register_op("ai.onnx", "Trilu")
def trilu(input: Tensor, k: Tensor = None, upper: int = 1) -> Tensor:
    """Compute Trilu."""
    inputs = [input]
    if k is not None:
        inputs.append(k)
    return record_op("Trilu", inputs, {"upper": upper})


@register_op("ai.onnx", "Unique")
def unique(X: Tensor, axis: int | None = None, sorted: int = 1) -> Tensor:
    """Compute Unique."""
    attrs = {"sorted": sorted}
    if axis is not None:
        attrs["axis"] = axis
    return record_op("Unique", [X], attrs)


@register_op("ai.onnx", "Bernoulli")
def bernoulli(input: Tensor, dtype: int | None = None, seed: float = 0.0) -> Tensor:
    """Compute Bernoulli."""
    attrs: dict[str, AttributeValue] = {"seed": seed}
    if dtype is not None:
        attrs["dtype"] = dtype
    return record_op("Bernoulli", [input], attrs)


@register_op("ai.onnx", "CenterCropPad")
def center_crop_pad(input_data: Tensor, shape: Tensor, axes: list[int] | None = None) -> Tensor:
    """Compute CenterCropPad."""
    attrs = {}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("CenterCropPad", [input_data, shape], attrs)


@register_op("ai.onnx", "Compress")
def compress(input: Tensor, condition: Tensor, axis: int | None = None) -> Tensor:
    """Compute Compress."""
    attrs = {}
    if axis is not None:
        attrs["axis"] = axis
    return record_op("Compress", [input, condition], attrs)


@register_op("ai.onnx", "Reshape")
def reshape(x: Tensor, shape: Tensor) -> Tensor:
    """Reshapes a tensor."""
    return record_op("Reshape", [x, shape])


@register_op("ai.onnx", "Transpose")
def transpose(x: Tensor, perm: list[int] | None = None) -> Tensor:
    """Transposes a tensor."""
    attributes = {"perm": perm} if perm is not None else {}
    return record_op("Transpose", [x], attributes)


@register_op("ai.onnx", "Expand")
def expand(input: Tensor, shape: Tensor) -> Tensor:
    """Execute the Expand process and return the computed results."""
    return record_op("Expand", [input, shape])


@register_op("ai.onnx", "Slice")
def slice(
    data: Tensor,
    starts: Tensor,
    ends: Tensor,
    axes: Tensor = None,
    steps: Tensor = None,
) -> Tensor:
    """Compute Slice."""
    inputs = [data, starts, ends]
    if axes is not None:
        inputs.append(axes)
        if steps is not None:
            inputs.append(steps)
    return record_op("Slice", inputs)


@register_op("ai.onnx", "Tile")
def tile(input: Tensor, repeats: Tensor) -> Tensor:
    """Compute Tile."""
    return record_op("Tile", [input, repeats])


@register_op("ai.onnx", "Gather")
def gather(data: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    """Compute Gather."""
    return record_op("Gather", [data, indices], {"axis": axis})


@register_op("ai.onnx", "GatherND")
def gather_nd(data: Tensor, indices: Tensor, batch_dims: int = 0) -> Tensor:
    """Compute GatherND."""
    return record_op("GatherND", [data, indices], {"batch_dims": batch_dims})


@register_op("ai.onnx", "GatherElements")
def gather_elements(data: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    """Compute GatherElements."""
    return record_op("GatherElements", [data, indices], {"axis": axis})


@register_op("ai.onnx", "DepthToSpace")
def depth_to_space(data: Tensor, blocksize: int, mode: str = "DCR") -> Tensor:
    """Compute DepthToSpace."""
    return record_op("DepthToSpace", [data], {"blocksize": blocksize, "mode": mode})


@register_op("ai.onnx", "SpaceToDepth")
def space_to_depth(data: Tensor, blocksize: int) -> Tensor:
    """Compute SpaceToDepth."""
    return record_op("SpaceToDepth", [data], {"blocksize": blocksize})


@register_op("ai.onnx", "OneHot")
def one_hot(indices: Tensor, depth: Tensor, values: Tensor, axis: int = -1) -> Tensor:
    """Compute OneHot."""
    return record_op("OneHot", [indices, depth, values], {"axis": axis})


@register_op("ai.onnx", "Flatten")
def flatten(x: Tensor) -> Tensor:
    """Compute Flatten."""
    return record_op("Flatten", [x])


@register_op("ai.onnx", "Identity")
def identity(x: Tensor) -> Tensor:
    """Compute Identity."""
    return record_op("Identity", [x])


@register_op("ai.onnx", "ImageDecoder")
def image_decoder(x: Tensor) -> Tensor:
    """Compute ImageDecoder."""
    return record_op("ImageDecoder", [x])


@register_op("ai.onnx", "Pad")
def pad(x: Tensor) -> Tensor:
    """Compute Pad."""
    return record_op("Pad", [x])


@register_op("ai.onnx", "Range")
def range_op2(x: Tensor) -> Tensor:
    """Compute Range."""
    return record_op("Range", [x])


@register_op("ai.onnx", "Shape")
def shape(x: Tensor) -> Tensor:
    """Compute Shape."""
    return record_op("Shape", [x])


@register_op("ai.onnx", "Split")
def split(x: Tensor) -> Tensor:
    """Compute Split."""
    return record_op("Split", [x])


@register_op("ai.onnx", "Squeeze")
def squeeze(x: Tensor) -> Tensor:
    """Compute Squeeze."""
    return record_op("Squeeze", [x])


@register_op("ai.onnx", "TensorScatter")
def tensor_scatter(x: Tensor) -> Tensor:
    """Compute TensorScatter."""
    return record_op("TensorScatter", [x])


@register_op("ai.onnx", "TfIdfVectorizer")
def tf_idf_vectorizer(x: Tensor) -> Tensor:
    """Compute TfIdfVectorizer."""
    return record_op("TfIdfVectorizer", [x])


@register_op("ai.onnx", "Unsqueeze")
def unsqueeze(x: Tensor) -> Tensor:
    """Compute Unsqueeze."""
    return record_op("Unsqueeze", [x])


@register_op("ai.onnx", "Upsample")
def upsample(x: Tensor) -> Tensor:
    """Compute Upsample."""
    return record_op("Upsample", [x])


@register_op("ai.onnx", "Repeat")
def repeat(x: Tensor, repeats: Tensor) -> Tensor:
    """Compute Repeat."""
    return record_op("Repeat", [x, repeats])
