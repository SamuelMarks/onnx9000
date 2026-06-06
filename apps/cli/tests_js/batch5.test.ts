import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { handleRocmCommand } from "../src/commands/rocm.js";
import { handleScriptCommand } from "../src/commands/script.js";
import { handleSimplifyCommand } from "../src/commands/simplify.js";
import { handleSKL2ONNX } from "../src/commands/skl2onnx.js";
import { handleSparseCommand } from "../src/commands/sparse.js";
import { handleSphinxDemoUICommand } from "../src/commands/sphinx-demo-ui.js";

import * as child_process from "child_process";
import * as fs from "fs";

vi.mock("child_process", () => ({
  spawn: vi.fn(),
}));

vi.mock("fs", () => ({
  default: {
    existsSync: vi.fn(),
  },
  existsSync: vi.fn(),
}));

describe("CLI Commands Batch 5", () => {
  let consoleLogSpy: any;
  let consoleErrorSpy: any;
  let processExitSpy: any;
  let processOnSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    processExitSpy = vi
      .spyOn(process, "exit")
      .mockImplementation(() => undefined as never);
    processOnSpy = vi
      .spyOn(process, "on")
      .mockImplementation(() => process as any);

    vi.mocked(fs.existsSync).mockReturnValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("handleRocmCommand", () => {
    it("shows help when no args or -h", () => {
      handleRocmCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleRocmCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleRocmCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Initializing ROCm execution for model.onnx"),
      );
      handleRocmCommand([""]);
    });
  });

  describe("handleScriptCommand", () => {
    it("shows help when no args or -h", () => {
      handleScriptCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleScriptCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command without output", () => {
      handleScriptCommand(["script.py"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Executing ONNX Script from script.py"),
      );
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          "Successfully compiled script. Use -o to save the output.",
        ),
      );
      handleScriptCommand([""]);
    });

    it("processes command with custom output -o", () => {
      handleScriptCommand(["script.py", "-o", "out.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Saved compiled ONNX to out.onnx"),
      );
      handleScriptCommand(["script.py", "-o"]);
    });
  });

  describe("handleSimplifyCommand", () => {
    it("shows help when no args or -h", () => {
      handleSimplifyCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleSimplifyCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command with default output", () => {
      handleSimplifyCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Saving simplified model to model_sim.onnx..."),
      );
      handleSimplifyCommand([""]);
    });

    it("processes command with custom output -o", () => {
      handleSimplifyCommand(["model.onnx", "-o", "out.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Saving simplified model to out.onnx..."),
      );
      handleSimplifyCommand(["model.onnx", "-o"]);
    });
  });

  describe("handleSKL2ONNX", () => {
    it("shows error when no args", () => {
      handleSKL2ONNX([]);
      expect(processExitSpy).toHaveBeenCalledWith(1);
    });

    it("processes command", () => {
      handleSKL2ONNX(["model.pkl"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("SKL2ONNX processed model.pkl"),
      );
    });
  });

  describe("handleSparseCommand", () => {
    it("shows help when no args or -h", () => {
      handleSparseCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleSparseCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    describe("prune command", () => {
      it("errors when missing model", () => {
        handleSparseCommand(["prune"]);
        expect(processExitSpy).toHaveBeenCalledWith(1);
        handleSparseCommand(["prune", "-x"]);
        expect(processExitSpy).toHaveBeenCalledWith(1);
      });

      it("processes with default output and sparsity", () => {
        handleSparseCommand(["prune", "model.onnx"]);
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining("Pruning model to 0.0 sparsity..."),
        );
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining(
            "Saving sparse model to model_sparse.onnx...",
          ),
        );
      });

      it("processes with custom output and sparsity", () => {
        handleSparseCommand([
          "prune",
          "model.onnx",
          "-o",
          "out.onnx",
          "--sparsity",
          "0.5",
        ]);
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining("Pruning model to 0.5 sparsity..."),
        );
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining("Saving sparse model to out.onnx..."),
        );
      });

      it("processes with recipe", () => {
        handleSparseCommand(["prune", "model.onnx", "--recipe", "recipe.yaml"]);
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining("Applying pruning recipe: recipe.yaml"),
        );
      });

      it("handles boundary flags", () => {
        handleSparseCommand(["prune", "model.onnx", "-o"]);
        handleSparseCommand(["prune", "model.onnx", "--sparsity"]);
        handleSparseCommand(["prune", "model.onnx", "--recipe"]);
      });

      it("handles undefined model fallback log", () => {
        // handle the line 42 model || ''
        const badModel = {
          startsWith: () => false,
          replace: () => "r",
          toString: () => "",
        } as any;
        Object.defineProperty(badModel, "length", { value: 1 });
        handleSparseCommand(["prune", badModel]);
      });
    });

    it("errors on unknown command", () => {
      handleSparseCommand(["unknown"]);
      expect(processExitSpy).toHaveBeenCalledWith(1);
      handleSparseCommand([""]);
    });
  });

  describe("handleSphinxDemoUICommand", () => {
    it("shows help when -h", async () => {
      await handleSphinxDemoUICommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("fails when ui dir not found", async () => {
      vi.mocked(fs.existsSync).mockReturnValue(false);
      try {
        await handleSphinxDemoUICommand([]);
      } catch (e) {}
      expect(processExitSpy).toHaveBeenCalledWith(1);
    });

    it("spawns pnpm dev when ui dir exists", async () => {
      vi.mocked(fs.existsSync).mockImplementation((p: any) => {
        if (p.includes("pnpm-workspace.yaml")) return true;
        if (p.includes("sphinx-demo-ui")) return true;
        return false;
      });

      let onCb: any;
      const mockChild = {
        on: vi.fn().mockImplementation((event, cb) => {
          if (event === "close") onCb = cb;
          return mockChild;
        }),
        kill: vi.fn(),
      };
      vi.mocked(child_process.spawn).mockReturnValue(mockChild as any);

      const promise = handleSphinxDemoUICommand([]);
      // we need to wait for imports to finish internally
      await new Promise((r) => setTimeout(r, 10));
      expect(child_process.spawn).toHaveBeenCalledWith(
        "pnpm",
        ["dev"],
        expect.any(Object),
      );

      onCb(0);
      await promise;
    });

    it("rejects when process exits with error", async () => {
      vi.mocked(fs.existsSync).mockReturnValue(true);

      let onCb: any;
      const mockChild = {
        on: vi.fn().mockImplementation((event, cb) => {
          if (event === "close") onCb = cb;
          return mockChild;
        }),
        kill: vi.fn(),
      };
      vi.mocked(child_process.spawn).mockReturnValue(mockChild as any);

      const promise = handleSphinxDemoUICommand([]);
      await new Promise((r) => setTimeout(r, 10));
      onCb(1);

      await expect(promise).rejects.toThrow(
        "Sphinx Demo UI exited with code 1",
      );
    });

    it("rejects on child error", async () => {
      vi.mocked(fs.existsSync).mockReturnValue(true);

      let errCb: any;
      const mockChild = {
        on: vi.fn().mockImplementation((event, cb) => {
          if (event === "error") errCb = cb;
          return mockChild;
        }),
        kill: vi.fn(),
      };
      vi.mocked(child_process.spawn).mockReturnValue(mockChild as any);

      const promise = handleSphinxDemoUICommand([]);
      await new Promise((r) => setTimeout(r, 10));
      errCb(new Error("child err"));

      await expect(promise).rejects.toThrow("child err");
    });

    it("handles SIGINT", async () => {
      vi.mocked(fs.existsSync).mockReturnValue(true);

      const mockChild = {
        on: vi.fn().mockReturnValue({}),
        kill: vi.fn(),
      };
      vi.mocked(child_process.spawn).mockReturnValue(mockChild as any);

      let sigintCb: any;
      processOnSpy.mockImplementation((event: string, cb: any) => {
        if (event === "SIGINT") sigintCb = cb;
        return process;
      });

      const promise = handleSphinxDemoUICommand([]);
      await new Promise((r) => setTimeout(r, 10));
      sigintCb();
      await promise;
      expect(mockChild.kill).toHaveBeenCalledWith("SIGINT");
    });
  });
});
