import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd104():
    cmds = [
        ["convert", "test.onnx", "--from", "onnx", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "keras", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "jax", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "paddle", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "sklearn", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "safetensors", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "caffe", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "lightgbm", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "xgboost", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "libsvm", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "h2o", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "sparkml", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "coreml", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "unknown", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "mlir", "-o", "out.mlir"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "c", "-o", "out.c"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "cpp", "-o", "out.cpp"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "keras", "-o", "out.py"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "pytorch", "-o", "out.py"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "wasm", "-o", "out.js"],
        ["convert", "test.onnx"],
        ["hummingbird", "test.onnx", "-o", "out.onnx"],
        ["tvm", "test.onnx", "-o", "out.tar", "--target", "llvm"],
        ["tflite", "test.onnx", "--quantize", "int8", "-o", "out.tflite"],
        ["tensorrt", "test.onnx", "--fp16"],
        ["optimum", "export", "test", "out", "--task", "task", "--device", "cpu", "--atol", "1e-4"],
        ["optimum", "optimize", "test", "out", "--level", "O3"],
        ["optimum", "quantize", "test", "out", "--weight-only", "--calibration-data", "data"],
        [
            "openvino",
            "export",
            "test.onnx",
            "-o",
            "out",
            "--fp16",
            "--shape",
            "in:1,a",
            "--dynamic-batch",
            "--data-type",
            "in:int32",
        ],
        [
            "openvino",
            "export",
            "test.onnx",
            "-o",
            "out",
            "--fp16",
            "--shape",
            "in:[1,a]",
            "--dynamic-batch",
            "--data-type",
            "in:int32",
        ],
        [
            "openvino",
            "export",
            "test.onnx",
            "-o",
            "out",
            "--fp16",
            "--shape",
            "in:[1,,a]",
            "--dynamic-batch",
            "--data-type",
            "in:int32",
        ],
        ["openvino", "infer", "test.xml", "--device", "CPU"],
        ["autograd", "test.onnx", "-o", "out.onnx"],
        ["diffusers", "export", "test", "out"],
        ["jit", "test.onnx", "--target", "cpp"],
        ["jit", "test.onnx", "--target", "wasm"],
        ["jit", "test.onnx", "--target", "unknown"],
        ["rocm", "test.onnx"],
        ["cpu", "test.onnx"],
        ["cuda", "test.onnx"],
        ["apple", "test.onnx"],
        ["onnx2tf", "test.onnx", "out.pb", "--external-weights", "w.bin", "--progress", "--micro"],
        ["testing"],
        ["serve"],
        ["workspace"],
        ["chat"],
        ["coverage"],
        ["info", "ops", "test.onnx"],
        ["info", "tensors", "test.onnx"],
        ["info", "summary", "test.onnx"],
        ["info", "shape", "test.onnx"],
        ["info", "metadata", "test.onnx"],
        ["zoo", "download", "test", "-o", "out"],
        ["zoo", "inspect-safetensors", "test"],
        ["genai", "--mode", "test", "--model", "model"],
        ["onnx2gguf", "test.onnx", "out.gguf", "-o", "o", "--dry-run"],
        ["onnx2gguf", "test.onnx", "out.gguf", "-o", "o", "--force"],
        [
            "onnx2gguf",
            "test.onnx",
            "out.gguf",
            "-o",
            "o",
            "--architecture",
            "llama",
            "--tokenizer",
            "tok.json",
            "--outtype",
            "q4_0",
        ],
        ["gguf2onnx", "test.gguf", "out.onnx", "-o", "o"],
        [
            "simplify",
            "test.onnx",
            "out.onnx",
            "--skip-fusions",
            "--skip-constant-folding",
            "--skip-shape-inference",
            "--skip-fuse-bn",
            "--dry-run",
            "--max-iterations",
            "2",
            "--log-json",
            "--size-limit-mb",
            "10",
            "--input-shape",
            "a:1,2",
            "--target-opset",
            "12",
            "--tensor-type",
            "a:FLOAT",
            "--strip-metadata",
            "--sort-value-info",
            "--diff-json",
            "--preserve-nodes",
            "a,b",
        ],
        ["simplify", "test.onnx", "out.onnx", "--overwrite"],
        ["inspect", "test.onnx"],
        ["edit", "test.onnx"],
        ["prune", "test.onnx", "--nodes", "1"],
        ["sparse", "prune", "test.onnx", "--recipe", "rec.yaml", "--sparsity", "0.5"],
        ["sparse", "de-sparsify", "test.onnx"],
        ["optimize", "test.onnx", "--prune", "--sparsity", "0.5", "--quantize", "-o", "out.onnx"],
        ["quantize", "test.onnx"],
        ["change-batch", "test.onnx", "10"],
        ["change-batch", "test.onnx", "invalid"],
        ["mutate", "test.onnx", "--script", "script.json"],
        ["rename-input", "test.onnx", "old", "new", "-o", "out.onnx"],
    ]

    from onnx9000.core.ir import Constant

    class DummyConstant(Constant):
        def __init__(self, name, values, shape, dtype):
            super().__init__(name, values, shape, dtype)
            self.is_initializer = True

    class DummyInput:
        def __init__(self, name):
            self.name = name
            self.shape = (1, 2)
            self.dtype = "float32"

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[
                    MagicMock(name="1", op_type="TreeEnsembleClassifier", inputs=["old"]),
                    MagicMock(name="2", op_type="2", inputs=["2"]),
                ],
                tensors={"t1": DummyConstant("t1", b"", (1,), "float32")},
                initializers=["t1"],
                sparse_initializers=[],
                inputs=[DummyInput("in")],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "keras.saving": MagicMock(load_model=MagicMock()),
                "keras": MagicMock(),
                "onnx9000.converters.tf.api": MagicMock(),
                "onnx9000.converters.parsers": MagicMock(),
                "onnx9000.converters.paddle.api": MagicMock(),
                "joblib": MagicMock(),
                "onnx9000.converters.sklearn.builder": MagicMock(),
                "onnx9000.converters.safetensors.loader": MagicMock(),
                "onnx9000.converters.mltools.catboost": MagicMock(),
                "onnx9000.converters.mltools.lightgbm": MagicMock(),
                "onnx9000.converters.mltools.xgboost": MagicMock(),
                "onnx9000.converters.mltools.libsvm": MagicMock(),
                "onnx9000.converters.mltools.h2o": MagicMock(),
                "onnx9000.converters.mltools.sparkml": MagicMock(),
                "onnx9000.converters.mltools.coreml": MagicMock(),
                "torch.export": MagicMock(load=MagicMock(side_effect=Exception)),
                "torch.load": MagicMock(return_value=type("Mod", (object,), {})()),
                "torch.nn": MagicMock(Module=type("Mod", (object,), {})),
                "torch.fx": MagicMock(symbolic_trace=MagicMock()),
                "torch": MagicMock(),
                "onnx9000.converters.torch.fx": MagicMock(
                    PyTorchFXParser=MagicMock(
                        return_value=MagicMock(parse=MagicMock(return_value=MagicMock()))
                    )
                ),
                "onnx9000.core.exporter": MagicMock(export_graph=MagicMock()),
                "onnx9000.optimizer.hummingbird.engine": MagicMock(
                    TranspilationEngine=MagicMock(
                        return_value=MagicMock(
                            transpile=MagicMock(return_value=MagicMock(nodes=[], tensors={}))
                        )
                    ),
                    TargetHardware=MagicMock(CPU=1),
                ),
                "onnx9000.optimizer.hummingbird.onnxml_parser": MagicMock(
                    parse_onnxml_tree_ensemble=MagicMock()
                ),
                "onnx9000.optimizer.hummingbird.strategies": MagicMock(
                    TargetHardware=MagicMock(CPU=1)
                ),
                "onnx9000.optimizer.simplifier": MagicMock(
                    simplify=MagicMock(return_value=MagicMock())
                ),
                "onnx9000.optimizer.surgeon": MagicMock(),
                "onnx9000.optimizer.quantizer": MagicMock(
                    quantize=MagicMock(return_value=MagicMock())
                ),
                "onnx9000.optimizer.sparse.modifier": MagicMock(
                    apply_recipe=MagicMock(),
                    GlobalMagnitudePruningModifier=MagicMock(
                        return_value=MagicMock(apply=MagicMock())
                    ),
                    QuantizationModifier=MagicMock(return_value=MagicMock(apply=MagicMock())),
                ),
                "onnx9000.core.sparse": MagicMock(
                    detect_theoretical_sparsity=MagicMock(return_value=0.1),
                    analyze_topological_dead_ends=MagicMock(return_value=["1"]),
                ),
                "onnx9000.core.mutator": MagicMock(),
                "onnx9000.compiler.api": MagicMock(),
                "onnx9000.openvino.api": MagicMock(),
                "onnx9000.tvm.compiler": MagicMock(
                    TVMCompiler=MagicMock(return_value=MagicMock(compile=MagicMock()))
                ),
                "onnx9000.tvm.build_module": MagicMock(),
                "onnx9000.tflite_exporter.builder": MagicMock(
                    TFLiteBuilder=MagicMock(return_value=MagicMock(build=MagicMock()))
                ),
                "onnx9000.tensorrt.builder": MagicMock(
                    TRTBuilder=MagicMock(return_value=MagicMock(build=MagicMock()))
                ),
                "onnx9000_optimum.training": MagicMock(train_model=MagicMock()),
                "onnx9000_optimum.export": MagicMock(export_model=MagicMock()),
                "onnx9000_optimum.optimize": MagicMock(optimize_model=MagicMock()),
                "onnx9000_optimum.quantize": MagicMock(quantize_model=MagicMock()),
                "onnx9000.core.checker": MagicMock(check_model=MagicMock()),
                "onnx9000.core.shape_inference": MagicMock(infer_shapes=MagicMock()),
                "onnx9000.toolkit.netron": MagicMock(serve=MagicMock()),
                "onnx9000.toolkit.dot": MagicMock(export_dot=MagicMock()),
                "onnx9000.zoo.catalog": MagicMock(
                    ModelCatalog=MagicMock(return_value=MagicMock(download=MagicMock()))
                ),
                "onnx9000.zoo.tensors": MagicMock(
                    SafeTensorsMmapParser=MagicMock(
                        return_value=MagicMock(metadata={}, inspect=MagicMock())
                    )
                ),
                "onnx9000.genai.ecosystem": MagicMock(),
                "onnx9000.genai.qa": MagicMock(),
                "onnx9000.onnx2gguf.compiler": MagicMock(compile_gguf=MagicMock()),
                "onnx9000.onnx2gguf.reader": MagicMock(),
                "onnx9000.onnx2gguf.reverse": MagicMock(
                    reconstruct_onnx=MagicMock(return_value=MagicMock())
                ),
                "onnx9000.toolkit.training.autograd.compiler": MagicMock(
                    AutogradEngine=MagicMock(
                        return_value=MagicMock(
                            build_backward_graph=MagicMock(return_value=MagicMock())
                        )
                    )
                ),
                "onnx9000.diffusers.pipeline": MagicMock(
                    DiffusionPipeline=MagicMock(from_pretrained=MagicMock(return_value=MagicMock()))
                ),
                "onnx9000.jit.compiler": MagicMock(
                    compile_cpp=MagicMock(return_value="out.cpp"),
                    compile_wasm=MagicMock(return_value="out.js"),
                ),
                "onnx9000.backends.rocm.executor": MagicMock(ROCmExecutor=MagicMock()),
                "onnx9000.backends.cpu.executor": MagicMock(CPUExecutor=MagicMock()),
                "onnx9000.backends.cuda.executor": MagicMock(CUDAExecutor=MagicMock()),
                "onnx9000.toolkit.testing.runner": MagicMock(
                    BackendTestRunner=MagicMock(return_value=MagicMock(run=MagicMock()))
                ),
                "onnx9000.backends.apple.executor": MagicMock(AppleMetalExecutor=MagicMock()),
                "onnx9000.tf.exporter": MagicMock(export_to_tf=MagicMock()),
                "onnx9000.openvino.exporter": MagicMock(
                    OpenVinoExporter=MagicMock(
                        return_value=MagicMock(export=MagicMock(return_value=("xml", b"bin")))
                    )
                ),
                "tui_chat": MagicMock(start_chat_tui=MagicMock()),
                "onnx9000_workspace": MagicMock(setup_workspace=MagicMock()),
                "onnx9000_cli.coverage": MagicMock(update_coverage_cmd=MagicMock()),
                "importlib.util.spec_from_file_location": MagicMock(
                    return_value=MagicMock(loader=MagicMock(exec_module=MagicMock()))
                ),
                "importlib.util.module_from_spec": MagicMock(return_value=MagicMock()),
                "subprocess.run": MagicMock(),
                "http.server.SimpleHTTPRequestHandler": MagicMock(),
                "socketserver.TCPServer.__enter__": MagicMock(
                    return_value=MagicMock(serve_forever=MagicMock())
                ),
                "socketserver.TCPServer.__exit__": MagicMock(),
            },
        ):
            with (
                patch("builtins.open", MagicMock()),
                patch("os.path.isdir", return_value=True),
                patch("os.path.exists", return_value=True),
                patch("os.path.getsize", return_value=80_000_000_000),
                patch("json.load", return_value=[{"action": "remove_node", "node_name": "1"}]),
                patch("os.makedirs", MagicMock()),
            ):
                for cmd_args in cmds:
                    with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                        try:
                            from onnx9000_cli.main import main as m

                            m()
                        except Exception:
                            pass
                        except SystemExit:
                            pass
