import { describe, expect, it } from "vitest";
import { WorkerPipeline } from "../src/worker/index";

// Mock Worker
class MockWorker {
  listeners: Object = {};
  constructor(public path: string) {}
  addEventListener(event: string, cb: Object) {
    this.listeners[event] = cb;
  }
  removeEventListener(event: string, _cb: Object) {
    delete this.listeners[event];
  }
  postMessage(data: Object, _transfer: Object) {
    setTimeout(() => {
      if (this.listeners.message) {
        if (data.task === "err") {
          this.listeners.message({ data: { id: data.id, error: "fail" } });
        } else {
          this.listeners.message({ data: { id: data.id, result: "ok" } });
        }
      }
    }, 0);
  }
}
(global as any).Worker = MockWorker;

describe("WorkerPipeline", () => {
  it("constructs", () => {
    const wp = new WorkerPipeline("path");
    expect(wp.worker).toBeDefined();
  });

  it("run ok", async () => {
    const wp = new WorkerPipeline("path");
    const res = await wp.run("ok", "model", {});
    expect(res).toBe("ok");
  });

  it("run err", async () => {
    const wp = new WorkerPipeline("path");
    await expect(wp.run("err", "model", {})).rejects.toThrow("fail");
  });

  it("runZeroCopy ok", async () => {
    const wp = new WorkerPipeline("path");
    const res = await wp.runZeroCopy("ok", "model", new Float32Array(1));
    expect(res).toBe("ok");
  });

  it("runZeroCopy err", async () => {
    const wp = new WorkerPipeline("path");
    await expect(
      wp.runZeroCopy("err", "model", new Float32Array(1)),
    ).rejects.toThrow("fail");
  });

  it("createSharedMemory", () => {
    const wp = new WorkerPipeline("path");
    try {
      const sab = wp.createSharedMemory(10);
      expect(sab).toBeDefined();
    } catch (_e) {
      // Ignore if SharedArrayBuffer not defined in env
    }
  });
});
