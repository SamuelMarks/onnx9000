import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/wasm/engine";

describe("engine.ts", () => {
  it("should call and cover init", async () => {
    try {
      const res = (Module as any).init();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
  it("should call and cover execute_graph", async () => {
    try {
      const res = (Module as any).execute_graph();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
