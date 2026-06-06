import * as fs from "node:fs";
import * as ireeCompiler from "@onnx9000/iree-compiler/src/cli.js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { handleIreeCommand } from "../src/commands/iree.js";
import { handleJaxCommand } from "../src/commands/jax.js";
import { handleJsonExtractCommand } from "../src/commands/json-extract.js";
import { handleKeras2ONNX } from "../src/commands/keras2onnx.js";
import { handleLlamaWebCommand } from "../src/commands/llama-web.js";
import { handleMlirCommand } from "../src/commands/mlir.js";

vi.mock("fs", () => ({
  default: {
    readFileSync: vi.fn(),
    writeFileSync: vi.fn(),
  },
  readFileSync: vi.fn(),
  writeFileSync: vi.fn(),
}));

vi.mock("@onnx9000/core", () => ({
  load: vi.fn().mockResolvedValue({
    data: new Uint8Array([1, 2, 3]),
    bigVal: 10n,
    str: "test",
  }),
  LLaMA: vi.fn(),
}));

vi.mock("@onnx9000/iree-compiler/src/cli.js", () => ({
  compileModel: vi.fn().mockResolvedValue(undefined),
}));

describe("CLI Commands Batch 3", () => {
  let consoleLogSpy: any;
  let _consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    _consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
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

  describe("handleIreeCommand", () => {
    it("shows help when no args or -h", async () => {
      await handleIreeCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      processExitSpy.mockClear();
      await handleIreeCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes compile subCommand", async () => {
      await handleIreeCommand(["compile", "model.onnx"]);
      expect(ireeCompiler.compileModel).toHaveBeenCalledWith(
        "model.onnx",
        expect.any(Object),
      );
      await handleIreeCommand(["compile"]);
      expect(ireeCompiler.compileModel).toHaveBeenCalledWith(
        "",
        expect.any(Object),
      );
    });

    it("processes run subCommand", async () => {
      await handleIreeCommand(["run", "model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Running model.onnx via IREE WVM..."),
      );
      await handleIreeCommand(["run"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Running  via IREE WVM..."),
      );
    });

    it("handles invalid subCommand", async () => {
      await handleIreeCommand(["invalid"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Invalid IREE command"),
      );
    });
  });

  describe("handleJaxCommand", () => {
    it("shows help when no args or -h", () => {
      handleJaxCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleJaxCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleJaxCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Converting JAX model model.onnx to ONNX"),
      );
      handleJaxCommand([""]);
    });
  });

  describe("handleJsonExtractCommand", () => {
    it("shows help when no args or -h", async () => {
      await handleJsonExtractCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      await handleJsonExtractCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes without output file", async () => {
      await handleJsonExtractCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Extracting JSON"),
      );
    });

    it("processes with output file -o", async () => {
      vi.mocked(fs.writeFileSync).mockClear();
      await handleJsonExtractCommand(["model.onnx", "-o", "out.json"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Extracted JSON written to out.json"),
      );
    });

    it("processes with output file --output", async () => {
      vi.mocked(fs.writeFileSync).mockClear();
      await handleJsonExtractCommand(["model.onnx", "--output", "out.json"]);
      await handleJsonExtractCommand(["model.onnx", "--output"]);
      await handleJsonExtractCommand(["model.onnx", "-o"]);
    });

    it("handles missing modelPath param fallbacks", async () => {
      vi.mocked(fs.writeFileSync).mockClear();
      await handleJsonExtractCommand(["", "-o", "out.json"]);
    });

    it("handles big ints correctly", async () => {
      const core = await import("@onnx9000/core");
      vi.mocked(core.load).mockResolvedValueOnce({
        bigVal: 10n,
        somethingElse: "value",
      });
      await handleJsonExtractCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("10n"),
      );
    });

    it("handles buffers correctly", async () => {
      const core = await import("@onnx9000/core");
      vi.mocked(core.load).mockResolvedValueOnce({
        data: new Uint8Array([1, 2, 3]),
      });
      await handleJsonExtractCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Buffer: 3 bytes"),
      );
    });
  });

  describe("handleKeras2ONNX", () => {
    it("shows error when no args", () => {
      handleKeras2ONNX([]);
      expect(processExitSpy).toHaveBeenCalledWith(1);
    });

    it("processes command", () => {
      handleKeras2ONNX(["model.keras"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Keras2ONNX processed model.keras"),
      );
    });
  });

  describe("handleLlamaWebCommand", () => {
    it("shows help when invalid args", () => {
      handleLlamaWebCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleLlamaWebCommand(["model", "--invalid"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleLlamaWebCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleLlamaWebCommand(["model", "not--prompt"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes without output file", () => {
      handleLlamaWebCommand(["model.onnx", "--prompt", "test"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Generated text:"),
      );
      handleLlamaWebCommand(["", "--prompt", "test"]);
    });

    it("processes with output file via -o", () => {
      vi.mocked(fs.writeFileSync).mockClear();
      handleLlamaWebCommand([
        "model.onnx",
        "--prompt",
        "test",
        "-o",
        "out.txt",
      ]);
      expect(fs.writeFileSync).toHaveBeenCalledWith(
        "out.txt",
        expect.any(String),
      );
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Output saved to out.txt"),
      );
    });

    it("processes with output file via --output", () => {
      vi.mocked(fs.writeFileSync).mockClear();
      handleLlamaWebCommand([
        "model.onnx",
        "--prompt",
        "test",
        "--output",
        "out.txt",
      ]);
      expect(fs.writeFileSync).toHaveBeenCalledWith(
        "out.txt",
        expect.any(String),
      );
      handleLlamaWebCommand(["model.onnx", "--prompt", "test", "-o"]);
      handleLlamaWebCommand(["model.onnx", "--prompt", "test", "--output"]);
      handleLlamaWebCommand(["model.onnx", "--prompt", ""]);
      handleLlamaWebCommand(["model.onnx", "--prompt", "test", "-o", ""]);
      handleLlamaWebCommand(["model.onnx", "--prompt", "test", "--output", ""]);
    });
  });

  describe("handleMlirCommand", () => {
    it("shows help when no args or -h", () => {
      handleMlirCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleMlirCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleMlirCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Lowering model.onnx to MLIR..."),
      );
      handleMlirCommand([""]);
    });
  });
});

import { handleMmdnnCommand } from "../src/commands/mmdnn.js";

describe("CLI Commands Batch 3 - MMDNN", () => {
  let consoleLogSpy: any;
  let _consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    _consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    processExitSpy = vi
      .spyOn(process, "exit")
      .mockImplementation(() => undefined as never);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("handleMmdnnCommand", () => {
    it("shows help when no args or -h", () => {
      handleMmdnnCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleMmdnnCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleMmdnnCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Converting model model.onnx via MMDNN"),
      );
      handleMmdnnCommand([""]);
    });
  });
});

import { handleMobileMemoryCommand } from "../src/commands/mobile-memory.js";

describe("CLI Commands Batch 3 - Mobile Memory", () => {
  let consoleLogSpy: any;
  let _consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    _consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    processExitSpy = vi
      .spyOn(process, "exit")
      .mockImplementation(() => undefined as never);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("handleMobileMemoryCommand", () => {
    it("shows help when no args or -h", () => {
      handleMobileMemoryCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleMobileMemoryCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleMobileMemoryCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          "Analyzing mobile memory usage for model.onnx...",
        ),
      );
      handleMobileMemoryCommand([""]);
    });
  });
});
