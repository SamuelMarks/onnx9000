import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { handleAgentCommand } from "../src/commands/agent.js";
import { handleAppleCommand } from "../src/commands/apple.js";
import { handleArena } from "../src/commands/arena.js";
import { handleAutogradCommand } from "../src/commands/autograd.js";
import { handleCoreMLCommand } from "../src/commands/coreml.js";

describe("CLI Commands Batch 1", () => {
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

  it("handleAgentCommand coverage", () => {
    handleAgentCommand([]);
    expect(processExitSpy).toHaveBeenCalledWith(0);
    processExitSpy.mockClear();
    handleAgentCommand(["-h"]);
    expect(processExitSpy).toHaveBeenCalledWith(0);
    processExitSpy.mockClear();
    handleAgentCommand(["my_task"]);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining("my_task"),
    );
  });

  it("handleAppleCommand coverage", () => {
    handleAppleCommand([]);
    expect(processExitSpy).toHaveBeenCalledWith(0);
    processExitSpy.mockClear();
    handleAppleCommand(["-h"]);
    expect(processExitSpy).toHaveBeenCalledWith(0);
    processExitSpy.mockClear();
    handleAppleCommand(["model.onnx"]);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining("model.onnx"),
    );
    // trigger args[0] || '' fallback
    handleAppleCommand([""]);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining("Loading model for Apple Metal execution: ..."),
    );
  });

  it("handleArena coverage", () => {
    handleArena([]);
    expect(processExitSpy).toHaveBeenCalledWith(1);
    processExitSpy.mockClear();
    handleArena(["model.onnx"]);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining("model.onnx"),
    );
  });

  it("handleAutogradCommand coverage", () => {
    handleAutogradCommand([]);
    expect(processExitSpy).toHaveBeenCalledWith(0);
    processExitSpy.mockClear();
    handleAutogradCommand(["-h"]);
    expect(processExitSpy).toHaveBeenCalledWith(0);
    processExitSpy.mockClear();
    handleAutogradCommand(["model.onnx"]);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining("model_bw.onnx"),
    );
    handleAutogradCommand(["model.onnx", "-o", "out.onnx"]);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining("out.onnx"),
    );
    // test the oIndex logic boundary and the model || '' fallback
    handleAutogradCommand(["-o"]);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining("-o"));
    // trigger the model[0] missing path
    handleAutogradCommand(["", "-o", "out.onnx"]);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining("out.onnx"),
    );
    // branch missing: oIndex !== -1 && oIndex + 1 >= args.length
    handleAutogradCommand(["model.onnx", "-o"]);
  });

  it("handleCoreMLCommand coverage", () => {
    handleCoreMLCommand([]);
    expect(processExitSpy).toHaveBeenCalledWith(0);
    processExitSpy.mockClear();
    handleCoreMLCommand(["-h"]);
    expect(processExitSpy).toHaveBeenCalledWith(0);
    processExitSpy.mockClear();
    handleCoreMLCommand(["model.onnx"]);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining("model.onnx"),
    );
    handleCoreMLCommand([""]);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining("Exporting ONNX model to CoreML/MIL: ..."),
    );
  });
});
