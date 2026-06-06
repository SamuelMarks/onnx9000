import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { __test_main__ } from "../src/index.js";
import * as mockConvert from "../src/commands/convert.js";
import * as mockInspect from "../src/commands/inspect.js";
import * as mockJsonExtract from "../src/commands/json-extract.js";
import * as mockPytorchCodegen from "../src/commands/pytorch-codegen.js";
import * as mockWhisperLlm from "../src/commands/whisper-llm.js";
import * as mockLlamaWeb from "../src/commands/llama-web.js";
import * as mockTfjsShim from "../src/commands/tfjs-shim.js";
import * as mockGguf from "../src/commands/gguf.js";
import * as mockTvm from "../src/commands/tvm.js";
import * as mockTensorRT from "../src/commands/tensorrt.js";
import * as mockDiffusers from "../src/commands/diffusers.js";
import * as mockServe from "@onnx9000/serve";
import * as mockIree from "../src/commands/iree.js";
import * as mockTriton from "../src/commands/triton.js";
import * as mockTransformers from "../src/commands/transformers.js";
import * as mockCoreML from "../src/commands/coreml.js";
import * as mockEdit from "../src/commands/edit.js";
import * as mockAutograd from "../src/commands/autograd.js";
import * as mockZoo from "../src/commands/zoo.js";
import * as mockHummingbird from "../src/commands/hummingbird.js";
import * as mockSparse from "../src/commands/sparse.js";
import * as mockOptimum from "../src/commands/optimum.js";
import * as mockSphinxDemoUi from "../src/commands/sphinx-demo-ui.js";
import * as mockExport from "../src/commands/export.js";
import * as mockOnnx2Tf from "../src/commands/onnx2tf.js";
import * as mockOptimize from "../src/commands/optimize.js";
import * as mockSimplify from "../src/commands/simplify.js";
import * as mockAgent from "../src/commands/agent.js";
import * as mockApple from "../src/commands/apple.js";
import * as mockOnnx2c from "../src/commands/onnx2c.js";
import * as mockCuda from "../src/commands/cuda.js";
import * as mockJax from "../src/commands/jax.js";
import * as mockMmdnn from "../src/commands/mmdnn.js";
import * as mockRocm from "../src/commands/rocm.js";
import * as mockWasm from "../src/commands/wasm.js";
import * as mockWebgpu from "../src/commands/webgpu.js";
import * as mockWebnnPolyfill from "../src/commands/webnn-polyfill.js";
import * as mockOnnxChecker from "../src/commands/onnx-checker.js";
import * as mockScript from "../src/commands/script.js";
import * as mockMlir from "../src/commands/mlir.js";
import * as mockMobileMemory from "../src/commands/mobile-memory.js";
import * as mockProgressiveLoading from "../src/commands/progressive-loading.js";
import * as mockTflite from "../src/commands/tflite.js";
import * as mockNewModelArch from "../src/commands/new-model-arch.js";
import * as mockZeroDepClassifier from "../src/commands/zero-dep-classifier.js";
import * as mockPaddle2ONNX from "../src/commands/paddle2onnx.js";
import * as mockKeras2ONNX from "../src/commands/keras2onnx.js";
import * as mockSKL2ONNX from "../src/commands/skl2onnx.js";
import * as mockArena from "../src/commands/arena.js";

vi.mock("../src/commands/convert.js", () => ({
  handleConvertCommand: vi.fn(),
}));
vi.mock("../src/commands/inspect.js", () => ({
  handleInspectCommand: vi.fn(),
}));
vi.mock("../src/commands/json-extract.js", () => ({
  handleJsonExtractCommand: vi.fn(),
}));
vi.mock("../src/commands/pytorch-codegen.js", () => ({
  handlePytorchCodegenCommand: vi.fn(),
}));
vi.mock("../src/commands/whisper-llm.js", () => ({
  handleWhisperLlmCommand: vi.fn(),
}));
vi.mock("../src/commands/llama-web.js", () => ({
  handleLlamaWebCommand: vi.fn(),
}));
vi.mock("../src/commands/tfjs-shim.js", () => ({
  handleTfjsShimCommand: vi.fn(),
}));
vi.mock("../src/commands/gguf.js", () => ({
  handleOnnx2GgufCommand: vi.fn(),
  handleGguf2OnnxCommand: vi.fn(),
}));
vi.mock("../src/commands/tvm.js", () => ({ handleTvmCommand: vi.fn() }));
vi.mock("../src/commands/tensorrt.js", () => ({
  handleTensorRTCommand: vi.fn(),
}));
vi.mock("../src/commands/diffusers.js", () => ({
  handleDiffusersCommand: vi.fn(),
}));
vi.mock("@onnx9000/serve", () => ({ runCli: vi.fn() }));
vi.mock("../src/commands/iree.js", () => ({ handleIreeCommand: vi.fn() }));
vi.mock("../src/commands/triton.js", () => ({ handleTritonCommand: vi.fn() }));
vi.mock("../src/commands/transformers.js", () => ({
  handleTransformersCommand: vi.fn(),
}));
vi.mock("../src/commands/coreml.js", () => ({ handleCoreMLCommand: vi.fn() }));
vi.mock("../src/commands/edit.js", () => ({ handleEditCommand: vi.fn() }));
vi.mock("../src/commands/autograd.js", () => ({
  handleAutogradCommand: vi.fn(),
}));
vi.mock("../src/commands/zoo.js", () => ({ handleZooCommand: vi.fn() }));
vi.mock("../src/commands/hummingbird.js", () => ({
  handleHummingbirdCommand: vi.fn(),
}));
vi.mock("../src/commands/sparse.js", () => ({ handleSparseCommand: vi.fn() }));
vi.mock("../src/commands/optimum.js", () => ({
  handleOptimumCommand: vi.fn(),
}));
vi.mock("../src/commands/sphinx-demo-ui.js", () => ({
  handleSphinxDemoUICommand: vi.fn(),
}));
vi.mock("../src/commands/export.js", () => ({ handleExportCommand: vi.fn() }));
vi.mock("../src/commands/onnx2tf.js", () => ({
  handleOnnx2TfCommand: vi.fn(),
}));
vi.mock("../src/commands/optimize.js", () => ({
  handleOptimizeCommand: vi.fn(),
}));
vi.mock("../src/commands/simplify.js", () => ({
  handleSimplifyCommand: vi.fn(),
}));
vi.mock("../src/commands/agent.js", () => ({ handleAgentCommand: vi.fn() }));
vi.mock("../src/commands/apple.js", () => ({ handleAppleCommand: vi.fn() }));
vi.mock("../src/commands/onnx2c.js", () => ({ handleOnnx2cCommand: vi.fn() }));
vi.mock("../src/commands/cuda.js", () => ({ handleCudaCommand: vi.fn() }));
vi.mock("../src/commands/jax.js", () => ({ handleJaxCommand: vi.fn() }));
vi.mock("../src/commands/mmdnn.js", () => ({ handleMmdnnCommand: vi.fn() }));
vi.mock("../src/commands/rocm.js", () => ({ handleRocmCommand: vi.fn() }));
vi.mock("../src/commands/wasm.js", () => ({ handleWasmCommand: vi.fn() }));
vi.mock("../src/commands/webgpu.js", () => ({ handleWebgpuCommand: vi.fn() }));
vi.mock("../src/commands/webnn-polyfill.js", () => ({
  handleWebnnPolyfillCommand: vi.fn(),
}));
vi.mock("../src/commands/onnx-checker.js", () => ({
  handleOnnxCheckerCommand: vi.fn(),
}));
vi.mock("../src/commands/script.js", () => ({ handleScriptCommand: vi.fn() }));
vi.mock("../src/commands/mlir.js", () => ({ handleMlirCommand: vi.fn() }));
vi.mock("../src/commands/mobile-memory.js", () => ({
  handleMobileMemoryCommand: vi.fn(),
}));
vi.mock("../src/commands/progressive-loading.js", () => ({
  handleProgressiveLoadingCommand: vi.fn(),
}));
vi.mock("../src/commands/tflite.js", () => ({ handleTfliteCommand: vi.fn() }));
vi.mock("../src/commands/new-model-arch.js", () => ({
  handleNewModelArchCommand: vi.fn(),
}));
vi.mock("../src/commands/zero-dep-classifier.js", () => ({
  handleZeroDepClassifierCommand: vi.fn(),
}));
vi.mock("../src/commands/paddle2onnx.js", () => ({
  handlePaddle2ONNX: vi.fn(),
}));
vi.mock("../src/commands/keras2onnx.js", () => ({ handleKeras2ONNX: vi.fn() }));
vi.mock("../src/commands/skl2onnx.js", () => ({ handleSKL2ONNX: vi.fn() }));
vi.mock("../src/commands/arena.js", () => ({ handleArena: vi.fn() }));

vi.mock("@onnx9000/array", () => ({
  default: {},
}));

vi.mock("@onnx9000/openvino-exporter", () => ({
  default: {},
}));

describe("CLI Index", () => {
  let originalArgv: string[];
  let processExitSpy: any;
  let consoleErrorSpy: any;

  beforeEach(() => {
    originalArgv = process.argv;
    processExitSpy = vi
      .spyOn(process, "exit")
      .mockImplementation(() => undefined as never);
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    process.argv = originalArgv;
    vi.restoreAllMocks();
  });

  const commands = [
    { name: "convert", mock: mockConvert.handleConvertCommand },
    { name: "inspect", mock: mockInspect.handleInspectCommand },
    { name: "json-extract", mock: mockJsonExtract.handleJsonExtractCommand },
    {
      name: "pytorch-codegen",
      mock: mockPytorchCodegen.handlePytorchCodegenCommand,
    },
    { name: "whisper-llm", mock: mockWhisperLlm.handleWhisperLlmCommand },
    { name: "llama-web", mock: mockLlamaWeb.handleLlamaWebCommand },
    { name: "tfjs-shim", mock: mockTfjsShim.handleTfjsShimCommand },
    { name: "onnx2gguf", mock: mockGguf.handleOnnx2GgufCommand },
    { name: "gguf2onnx", mock: mockGguf.handleGguf2OnnxCommand },
    { name: "tvm", mock: mockTvm.handleTvmCommand },
    { name: "tensorrt", mock: mockTensorRT.handleTensorRTCommand },
    { name: "diffusers", mock: mockDiffusers.handleDiffusersCommand },
    { name: "iree", mock: mockIree.handleIreeCommand },
    { name: "triton", mock: mockTriton.handleTritonCommand },
    { name: "transformers", mock: mockTransformers.handleTransformersCommand },
    { name: "coreml", mock: mockCoreML.handleCoreMLCommand },
    { name: "edit", mock: mockEdit.handleEditCommand },
    { name: "autograd", mock: mockAutograd.handleAutogradCommand },
    { name: "zoo", mock: mockZoo.handleZooCommand },
    { name: "hummingbird", mock: mockHummingbird.handleHummingbirdCommand },
    { name: "sparse", mock: mockSparse.handleSparseCommand },
    { name: "optimum", mock: mockOptimum.handleOptimumCommand },
    {
      name: "sphinx-demo-ui",
      mock: mockSphinxDemoUi.handleSphinxDemoUICommand,
    },
    { name: "export", mock: mockExport.handleExportCommand },
    { name: "onnx2tf", mock: mockOnnx2Tf.handleOnnx2TfCommand },
    { name: "optimize", mock: mockOptimize.handleOptimizeCommand },
    { name: "simplify", mock: mockSimplify.handleSimplifyCommand },
    { name: "agent", mock: mockAgent.handleAgentCommand },
    { name: "apple", mock: mockApple.handleAppleCommand },
    { name: "onnx2c", mock: mockOnnx2c.handleOnnx2cCommand },
    { name: "cuda", mock: mockCuda.handleCudaCommand },
    { name: "jax", mock: mockJax.handleJaxCommand },
    { name: "mmdnn", mock: mockMmdnn.handleMmdnnCommand },
    { name: "rocm", mock: mockRocm.handleRocmCommand },
    { name: "wasm", mock: mockWasm.handleWasmCommand },
    { name: "webgpu", mock: mockWebgpu.handleWebgpuCommand },
    {
      name: "webnn-polyfill",
      mock: mockWebnnPolyfill.handleWebnnPolyfillCommand,
    },
    { name: "onnx-checker", mock: mockOnnxChecker.handleOnnxCheckerCommand },
    { name: "script", mock: mockScript.handleScriptCommand },
    { name: "mlir", mock: mockMlir.handleMlirCommand },
    { name: "mobile-memory", mock: mockMobileMemory.handleMobileMemoryCommand },
    {
      name: "progressive-loading",
      mock: mockProgressiveLoading.handleProgressiveLoadingCommand,
    },
    { name: "tflite", mock: mockTflite.handleTfliteCommand },
    {
      name: "new-model-arch",
      mock: mockNewModelArch.handleNewModelArchCommand,
    },
    {
      name: "zero-dep-classifier",
      mock: mockZeroDepClassifier.handleZeroDepClassifierCommand,
    },
    { name: "paddle2onnx", mock: mockPaddle2ONNX.handlePaddle2ONNX },
    { name: "keras2onnx", mock: mockKeras2ONNX.handleKeras2ONNX },
    { name: "skl2onnx", mock: mockSKL2ONNX.handleSKL2ONNX },
    { name: "arena", mock: mockArena.handleArena },
  ];

  for (const cmd of commands) {
    it(`handles ${cmd.name} command`, async () => {
      process.argv = ["node", "index.js", cmd.name, "arg1", "arg2"];
      await __test_main__();
      expect(cmd.mock).toHaveBeenCalledWith(["arg1", "arg2"]);
    });
  }

  it("handles serve command", async () => {
    process.argv = ["node", "index.js", "serve", "arg1"];
    await __test_main__();
    expect(mockServe.runCli).toHaveBeenCalledWith(["arg1"]);
  });

  it("handles array command module dynamic load", async () => {
    process.argv = ["node", "index.js", "array"];
    await __test_main__();
  });

  it("handles openvino command module dynamic load", async () => {
    process.argv = ["node", "index.js", "openvino"];
    await __test_main__();
  });

  it("handles unknown command", async () => {
    process.argv = ["node", "index.js", "unknown_cmd"];
    await __test_main__();
    expect(processExitSpy).toHaveBeenCalledWith(1);
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.stringContaining("Usage: onnx9000 <command>"),
    );
  });
});
