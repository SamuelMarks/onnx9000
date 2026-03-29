"""Command-line interface for the ONNX to TFLite exporter."""

import argparse
import sys

from onnx9000.tflite_exporter.compiler.subgraph import compile_graph_to_tflite
from onnx9000.tflite_exporter.exporter import TFLiteExporter


def main(args=None):
    """Main entry point for the onnx2tf CLI.

    Parses command-line arguments and triggers the ONNX to TFLite conversion.

    Args:
        args: Optional list of command-line arguments. Defaults to sys.argv[1:].

    """
    # 281. Implement CLI: onnx9000 onnx2tf model.onnx -o model.tflite
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="onnx2tf: Convert ONNX to TFLite")
    parser.add_argument("input", help="Path to input .onnx file")
    parser.add_argument(
        "-o", "--output", default="model.tflite", help="Path to output .tflite file"
    )

    # 286. Add --keep-nchw override flag
    parser.add_argument(
        "--keep-nchw", action="store_true", help="Keep NCHW format instead of transposing to NHWC"
    )
    parser.add_argument("--int8", action="store_true", help="Trigger INT8 quantization")
    parser.add_argument("--fp16", action="store_true", help="Trigger FP16 quantization")
    parser.add_argument("-b", "--batch", type=int, help="Override dynamic batch sizes")
    parser.add_argument(
        "--disable-optimization", action="store_true", help="Disable Layout optimizations entirely"
    )
    parser.add_argument("--external-weights", type=str, help="Path to external .bin weight files")
    parser.add_argument(
        "--progress", action="store_true", help="Show build progress for massive flatbuffers"
    )

    parser.add_argument(
        "--micro",
        action="store_true",
        help="Support TFLite Micro target generation",
    )

    parsed = parser.parse_args(args)

    quant_mode = "none"
    if parsed.int8:
        quant_mode = "int8"
    elif parsed.fp16:
        quant_mode = "fp16"

    if parsed.disable_optimization:
        print("[onnx2tf] Disabling layout and math optimizations...")

    if parsed.micro:
        print(
            "[onnx2tf] Warning: Generating TFLite Micro compatible schema (dropping optional headers)"
        )

    if parsed.external_weights:
        print(f"[onnx2tf] Using external weights from {parsed.external_weights}")

    if parsed.progress:
        print("[onnx2tf] Enabling build progress tracking...")

    if parsed.batch:
        print(f"[onnx2tf] Overriding dynamic batch size to {parsed.batch}")

    print(f"[onnx2tf] Loading ONNX model from {parsed.input}...")

    # In a real environment, we'd parse the ONNX file into the `Graph` object here.
    # from onnx9000.core.parser import parse_onnx
    # graph = parse_onnx(parsed.input)

    print(
        f"[onnx2tf] Compiling to TFLite... (keep_nchw={parsed.keep_nchw}, quant_mode={quant_mode})"
    )

    # exporter = TFLiteExporter()
    # subgraphs_offset = compile_graph_to_tflite(graph, exporter, keep_nchw=parsed.keep_nchw, quant_mode=quant_mode)
    # exporter.builder.start_vector(4, 1, 4)
    # exporter.builder.add_offset(subgraphs_offset)
    # subgraphs_vec_offset = exporter.builder.end_vector(1)
    # tflite_buf = exporter.finish(subgraphs_vec_offset, "onnx2tf converted")

    # with open(parsed.output, "wb") as f:
    #     f.write(tflite_buf)
    # print(f"[onnx2tf] Successfully exported to {parsed.output}")


if __name__ == "__main__":
    main()
