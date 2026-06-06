import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { handleNewModelArchCommand } from "../src/commands/new-model-arch.js";
import { handleOnnxCheckerCommand } from "../src/commands/onnx-checker.js";
import { handleOnnx2cCommand } from "../src/commands/onnx2c.js";
import { handleOnnx2TfCommand } from "../src/commands/onnx2tf.js";
import { handleOptimizeCommand } from "../src/commands/optimize.js";
import { handleOptimumCommand } from "../src/commands/optimum.js";
import { handlePaddle2ONNX } from "../src/commands/paddle2onnx.js";
import { handleProgressiveLoadingCommand } from "../src/commands/progressive-loading.js";
import { handlePytorchCodegenCommand } from "../src/commands/pytorch-codegen.js";

import * as fs from "fs";
import * as core from "@onnx9000/core";

vi.mock("fs", () => ({
  default: {
    readFileSync: vi.fn(),
    writeFileSync: vi.fn(),
  },
  readFileSync: vi.fn(),
  writeFileSync: vi.fn(),
}));

vi.mock("@onnx9000/core", () => {
  return {
    load: vi.fn().mockResolvedValue({}),
    ONNXToPyTorchVisitor: class {
      generate() {
        return "pytorch_code_mock";
      }
    },
  };
});

describe("CLI Commands Batch 4", () => {
  let consoleLogSpy: any;
  let consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    processExitSpy = vi
      .spyOn(process, "exit")
      .mockImplementation(() => undefined as never);

    vi.mocked(fs.readFileSync).mockReturnValue({
      buffer: new Uint8Array([1, 2, 3]).buffer,
    } as any);
    vi.mocked(fs.writeFileSync).mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("handleNewModelArchCommand", () => {
    it("shows help when no args or -h", () => {
      handleNewModelArchCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleNewModelArchCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleNewModelArchCommand(["--help"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleNewModelArchCommand(["my_arch"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          "Scaffolding new model architecture for: my_arch...",
        ),
      );
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("- src/models/my_arch/model.py"),
      );
      handleNewModelArchCommand([""]);
    });
  });

  describe("handleOnnxCheckerCommand", () => {
    it("shows help when no args or -h", () => {
      handleOnnxCheckerCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleOnnxCheckerCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleOnnxCheckerCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Checking ONNX model model.onnx..."),
      );
      handleOnnxCheckerCommand([""]);
    });
  });

  describe("handleOnnx2cCommand", () => {
    it("shows help when no args or -h", () => {
      handleOnnx2cCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleOnnx2cCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command with default output", () => {
      handleOnnx2cCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Successfully generated C code to output.c"),
      );
      handleOnnx2cCommand([""]);
    });

    it("processes command with custom output -o", () => {
      handleOnnx2cCommand(["model.onnx", "-o", "custom.c"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Successfully generated C code to custom.c"),
      );
      handleOnnx2cCommand(["model.onnx", "-o"]);
    });
  });

  describe("handleOnnx2TfCommand", () => {
    it("shows help when no args or -h", () => {
      handleOnnx2TfCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleOnnx2TfCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command with default output", () => {
      handleOnnx2TfCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Saving TFLite model to model.tflite..."),
      );
      expect(consoleLogSpy).not.toHaveBeenCalledWith(
        expect.stringContaining("INT8 quantization"),
      );
      handleOnnx2TfCommand([""]);
    });

    it("processes command with custom output -o", () => {
      handleOnnx2TfCommand(["model.onnx", "-o", "custom.tflite"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Saving TFLite model to custom.tflite..."),
      );
      handleOnnx2TfCommand(["model.onnx", "-o"]);
    });

    it("processes command with --int8", () => {
      handleOnnx2TfCommand(["model.onnx", "--int8"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("INT8 quantization"),
      );
    });
  });

  describe("handleOptimizeCommand", () => {
    it("shows help when no args or -h", () => {
      handleOptimizeCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleOptimizeCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command with default output", () => {
      handleOptimizeCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Saving optimized model to model_opt.onnx..."),
      );
      handleOptimizeCommand([""]);
    });

    it("processes command with custom output -o", () => {
      handleOptimizeCommand(["model.onnx", "-o", "custom.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Saving optimized model to custom.onnx..."),
      );
      handleOptimizeCommand(["model.onnx", "-o"]);
    });

    it("processes command with passes --passes", () => {
      handleOptimizeCommand(["model.onnx", "--passes", "p1,p2"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Running optimization passes: p1,p2"),
      );
      handleOptimizeCommand(["model.onnx", "--passes"]);
    });
  });

  describe("handleOptimumCommand", () => {
    it("shows help when no args or -h", () => {
      handleOptimumCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleOptimumCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    describe("export command", () => {
      it("errors when missing model_id", () => {
        handleOptimumCommand(["export"]);
        expect(processExitSpy).toHaveBeenCalledWith(1);
        handleOptimumCommand(["export", "-x"]);
        expect(processExitSpy).toHaveBeenCalledWith(1);
      });

      it("processes export with task", () => {
        handleOptimumCommand(["export", "model123", "--task", "my-task"]);
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining("Exporting model123 for task my-task"),
        );
      });

      it("processes export without task", () => {
        handleOptimumCommand(["export", "model123"]);
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining("Exporting model123 for task default"),
        );
        handleOptimumCommand(["export", "model123", "--task"]);
      });

      it("processes boundary logic when model is empty", () => {
        // handle empty string by using a truthy value that does not start with '-' but resolves to empty in console.log
        handleOptimumCommand([
          "export",
          { startsWith: () => false, toString: () => "" } as any,
        ]);
        handleOptimumCommand([
          "export",
          { startsWith: () => false, toString: () => "" } as any,
          "--task",
          { toString: () => "" } as any,
        ]);
      });
    });

    describe("optimize command", () => {
      it("errors when missing model", () => {
        handleOptimumCommand(["optimize"]);
        expect(processExitSpy).toHaveBeenCalledWith(1);
        handleOptimumCommand(["optimize", "-x"]);
        expect(processExitSpy).toHaveBeenCalledWith(1);
      });

      it("processes optimize with level and size", () => {
        handleOptimumCommand([
          "optimize",
          "model.onnx",
          "--level",
          "3",
          "--optimize-size",
        ]);
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining(
            "Optimizing model.onnx at level 3 for size...",
          ),
        );
      });

      it("processes optimize without level", () => {
        handleOptimumCommand(["optimize", "model.onnx"]);
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining("Optimizing model.onnx at level 1..."),
        );
        handleOptimumCommand(["optimize", "model.onnx", "--level"]);
      });

      it("processes boundary logic when model is empty", () => {
        handleOptimumCommand([
          "optimize",
          { startsWith: () => false, toString: () => "" } as any,
        ]);
        handleOptimumCommand([
          "optimize",
          "model.onnx",
          "--level",
          { toString: () => "" } as any,
        ]);
        handleOptimumCommand([
          "optimize",
          { startsWith: () => false, toString: () => "" } as any,
          "--level",
          { toString: () => "" } as any,
        ]);
      });
    });

    describe("quantize command", () => {
      it("errors when missing model", () => {
        handleOptimumCommand(["quantize"]);
        expect(processExitSpy).toHaveBeenCalledWith(1);
        handleOptimumCommand(["quantize", "-x"]);
        expect(processExitSpy).toHaveBeenCalledWith(1);
      });

      it("processes quantize with type", () => {
        handleOptimumCommand(["quantize", "model.onnx", "--quantize", "gptq"]);
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining("Quantizing model.onnx with method gptq..."),
        );
      });

      it("processes quantize without type", () => {
        handleOptimumCommand(["quantize", "model.onnx"]);
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining(
            "Quantizing model.onnx with method dynamic...",
          ),
        );
        handleOptimumCommand(["quantize", "model.onnx", "--quantize"]);
      });

      it("processes boundary logic when model is empty", () => {
        handleOptimumCommand([
          "quantize",
          { startsWith: () => false, toString: () => "" } as any,
        ]);
        handleOptimumCommand([
          "quantize",
          "model.onnx",
          "--quantize",
          { toString: () => "" } as any,
        ]);
        handleOptimumCommand([
          "quantize",
          { startsWith: () => false, toString: () => "" } as any,
          "--quantize",
          { toString: () => "" } as any,
        ]);
      });
    });

    it("errors on unknown command", () => {
      handleOptimumCommand(["unknown"]);
      expect(processExitSpy).toHaveBeenCalledWith(1);
      handleOptimumCommand([
        { startsWith: () => false, toString: () => "" } as any,
      ]);
      handleOptimumCommand([""]);
    });
  });

  describe("handlePaddle2ONNX", () => {
    it("shows error when no args", () => {
      handlePaddle2ONNX([]);
      expect(processExitSpy).toHaveBeenCalledWith(1);
    });

    it("processes command", () => {
      handlePaddle2ONNX(["model.paddle"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Paddle2ONNX processed model.paddle"),
      );
    });
  });

  describe("handleProgressiveLoadingCommand", () => {
    it("shows help when no args or -h", () => {
      handleProgressiveLoadingCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleProgressiveLoadingCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleProgressiveLoadingCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          "Generating progressive loading chunks for model.onnx...",
        ),
      );
      handleProgressiveLoadingCommand([""]);
    });
  });

  describe("handlePytorchCodegenCommand", () => {
    it("shows help when no args or -h", async () => {
      await handlePytorchCodegenCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      await handlePytorchCodegenCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes without output file", async () => {
      await handlePytorchCodegenCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Generating PyTorch code from model.onnx..."),
      );
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("pytorch_code_mock"),
      );
    });

    it("processes with output file -o", async () => {
      vi.mocked(fs.writeFileSync).mockClear();
      await handlePytorchCodegenCommand(["model.onnx", "-o", "out.py"]);
      expect(fs.writeFileSync).toHaveBeenCalledWith(
        "out.py",
        "pytorch_code_mock",
      );
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("PyTorch code written to out.py"),
      );
    });

    it("processes with output file --output", async () => {
      vi.mocked(fs.writeFileSync).mockClear();
      await handlePytorchCodegenCommand(["model.onnx", "--output", "out.py"]);
      expect(fs.writeFileSync).toHaveBeenCalledWith(
        "out.py",
        "pytorch_code_mock",
      );
      await handlePytorchCodegenCommand(["model.onnx", "--output"]);
      await handlePytorchCodegenCommand(["model.onnx", "-o"]);
      await handlePytorchCodegenCommand([""]);
    });
  });
});
