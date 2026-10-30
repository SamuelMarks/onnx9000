"""Keras preprocessing, augmentation, audio, and lookup layer converters."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_keras_initializers_stft(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Initializers Stft operation to ONNX equivalent."""
    return _map_keras_stft(builder, node)


def _map_keras_initializers_stft_initializer(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Initializers Stft Initializer operation to ONNX equivalent."""
    return _map_keras_stft(builder, node)


def _map_keras_initializers_stft_ext(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Initializers Stft Ext operation to ONNX equivalent."""
    return _map_keras_stft(builder, node)


def _map_keras_layers__category_encoding(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Category Encoding operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__discretization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Discretization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__hashed_crossing(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Hashed Crossing operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__hashing(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Hashing operation to ONNX equivalent."""
    return _map_keras_hashing(builder, node)


def _map_keras_layers__aug_mix(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Aug Mix operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__auto_contrast(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Auto Contrast operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__center_crop(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Center Crop operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__cut_mix(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Cut Mix operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__equalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Equalization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_num_bounding_boxes(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Max Num Bounding Boxes operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__mix_up(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Mix Up operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__rand_augment(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Rand Augment operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_brightness(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Brightness operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_color_degeneration(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Random Color Degeneration operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_color_jitter(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Random Color Jitter operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_contrast(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Contrast operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_crop(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Crop operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_elastic_transform(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Random Elastic Transform operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_erasing(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Erasing operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_flip(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Flip operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_gaussian_blur(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Random Gaussian Blur operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_grayscale(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Grayscale operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_hue(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Hue operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_invert(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Invert operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_perspective(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Perspective operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_posterization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Random Posterization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_rotation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Rotation operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_saturation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Saturation operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_sharpness(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Sharpness operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_shear(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Shear operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_translation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Translation operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_zoom(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Random Zoom operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__resizing(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Resizing operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__solarization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Solarization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__integer_lookup(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Integer Lookup operation to ONNX equivalent."""
    return _map_keras_integer_lookup(builder, node)


def _map_keras_mel_spectrogram(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Mel Spectrogram operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    fft_length = attr.pop("fft_length", 2048)
    stride = attr.pop("sequence_stride", 512)
    window_length = attr.pop("sequence_length", fft_length)
    frame_length_node = builder.make_node(
        "Constant", [], {"value": [window_length]}, node.name + "_frame_length"
    )[0]
    frame_step_node = builder.make_node(
        "Constant", [], {"value": [stride]}, node.name + "_frame_step"
    )[0]
    stft_out = builder.make_node(
        "STFT",
        [node.inputs[0], frame_step_node, frame_length_node],
        attr,
        node.name + "_stft",
    )[0]
    mag = builder.make_node("Abs", [stft_out], {}, node.name + "_mag")[0]
    mel_weights = builder.make_node("Constant", [], {"value": []}, node.name + "_mel_weights")[0]
    return builder.make_node("MatMul", [mag, mel_weights], {}, node.name)


def _map_keras_layers__normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Normalization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__pipeline(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Pipeline operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__rescaling(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Rescaling operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_stft_spectrogram(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Stft Spectrogram operation to ONNX equivalent."""
    return _map_keras_stft(builder, node)


def _map_keras_layers__string_lookup(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  String Lookup operation to ONNX equivalent."""
    return _map_keras_string_lookup(builder, node)


def _map_keras_text_vectorization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Text Vectorization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    return builder.make_node("Identity", node.inputs[:1], attr, node.name)


def _map_keras_string_lookup(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras String Lookup operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    return builder.make_node("CategoryMapper", node.inputs, attr, node.name)


def _map_keras_integer_lookup(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Integer Lookup operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    return builder.make_node("CategoryMapper", node.inputs, attr, node.name)


def _map_keras_hashing(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Hashing operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    return builder.make_node("CategoryMapper", node.inputs, attr, node.name)


def _map_keras_stft(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Stft operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    frame_length = attr.pop("sequence_length", None)
    if frame_length is None:
        frame_length = attr.pop("frame_length", 2048)
    frame_step = attr.pop("sequence_stride", None)
    if frame_step is None:
        frame_step = attr.pop("frame_step", 512)
    frame_length_node = builder.make_node(
        "Constant", [], {"value": [frame_length]}, node.name + "_frame_length"
    )[0]
    frame_step_node = builder.make_node(
        "Constant", [], {"value": [frame_step]}, node.name + "_frame_step"
    )[0]
    return builder.make_node(
        "STFT", [node.inputs[0], frame_step_node, frame_length_node], attr, node.name
    )


def _map_keras_layers__text_vectorization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Text Vectorization operation to ONNX equivalent."""
    return _map_keras_text_vectorization(builder, node)


def _map_keras_layers__mel_spectrogram(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Mel Spectrogram operation to ONNX equivalent."""
    return _map_keras_mel_spectrogram(builder, node)


def _map_keras_mfcc(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Mfcc operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    return builder.make_node("Identity", node.inputs[:1], attr, node.name)
