import * as transformers from "@onnx9000/transformers";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { handleTransformersCommand } from "../src/commands/transformers.js";

vi.mock("@onnx9000/transformers", () => ({
  pipeline: vi.fn(),
}));

describe("CLI Commands Batch 5", () => {
  let consoleLogSpy: any;
  let consoleErrorSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("handleTransformersCommand", () => {
    it("shows help when no args or -h", async () => {
      await handleTransformersCommand([]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Usage:"),
      );
      await handleTransformersCommand(["-h"]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining("Usage:"),
      );
    });

    it("processes command successfully", async () => {
      const mockPipe = vi
        .fn()
        .mockResolvedValue({ label: "POSITIVE", score: 0.99 });
      vi.mocked(transformers.pipeline).mockResolvedValue(mockPipe as any);

      await handleTransformersCommand(["text-classification", "test", "input"]);
      expect(transformers.pipeline).toHaveBeenCalledWith("text-classification");
      expect(mockPipe).toHaveBeenCalledWith("test input");
      expect(consoleLogSpy).toHaveBeenCalledWith("Result:", expect.any(String));
    });

    it("processes command with default input string", async () => {
      const mockPipe = vi
        .fn()
        .mockResolvedValue({ label: "POSITIVE", score: 0.99 });
      vi.mocked(transformers.pipeline).mockResolvedValue(mockPipe as any);

      await handleTransformersCommand(["text-classification"]);
      expect(mockPipe).toHaveBeenCalledWith("I love ONNX9000!");
    });

    it("handles errors gracefully", async () => {
      vi.mocked(transformers.pipeline).mockRejectedValue(
        new Error("Init failed"),
      );

      await handleTransformersCommand(["text-classification"]);
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        "Pipeline execution failed:",
        "Init failed",
      );
    });
  });
});
