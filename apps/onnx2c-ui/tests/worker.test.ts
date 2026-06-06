import { describe, it, expect, vi } from "vitest";
import { handleWorkerMessage } from "../src/worker";

vi.mock("@onnx9000/c-compiler", () => ({
  compileOnnxToC: vi.fn().mockImplementation((buffer, options) => {
    if (buffer === "error") return Promise.reject(new Error("Compile failed"));
    return Promise.resolve({
      header: "mock header",
      source: "mock source",
      summary: "mock summary",
    });
  }),
}));

describe("handleWorkerMessage", () => {
  it("should process valid request successfully", async () => {
    const postMessage = vi.fn();
    const event = {
      data: { buffer: new Uint8Array([1, 2, 3]), options: {} },
    } as any;

    await handleWorkerMessage(event, postMessage);

    expect(postMessage).toHaveBeenCalledWith({
      header: "mock header",
      source: "mock source",
      summary: "mock summary",
      arenaSize: 250000,
    });
  });

  it("should handle compilation errors", async () => {
    const postMessage = vi.fn();
    const event = {
      data: { buffer: "error", options: {} },
    } as any;

    await handleWorkerMessage(event, postMessage);

    expect(postMessage).toHaveBeenCalledWith({
      error: "Compile failed",
    });
  });
});
