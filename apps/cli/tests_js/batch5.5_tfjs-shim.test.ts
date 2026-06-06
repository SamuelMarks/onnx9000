import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { handleTfjsShimCommand } from "../src/commands/tfjs-shim.js";
import { handleTritonCommand } from "../src/commands/triton.js";
import { handleTvmCommand } from "../src/commands/tvm.js";

describe("CLI Commands Batch 5", () => {
  let consoleLogSpy: any;
  let consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    processExitSpy = vi
      .spyOn(process, "exit")
      .mockImplementation(() => undefined as never);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("handleTfjsShimCommand", () => {
    it("shows help when -h or --help", () => {
      handleTfjsShimCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleTfjsShimCommand(["--help"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleTfjsShimCommand([]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Testing TFJS Shim compatibility..."),
      );
    });
  });

  describe("handleTritonCommand", () => {
    it("shows help when no args or -h", () => {
      handleTritonCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleTritonCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleTritonCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Generating Triton code from model.onnx..."),
      );
      handleTritonCommand([""]);
    });
  });

  describe("handleTvmCommand", () => {
    it("shows help when no args or -h", () => {
      handleTvmCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleTvmCommand(["-h"]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it("processes command", () => {
      handleTvmCommand(["model.onnx"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("TVM compiling model.onnx for webgpu"),
      );
      handleTvmCommand([""]);
    });
  });
});
