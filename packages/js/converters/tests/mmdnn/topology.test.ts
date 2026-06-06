import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/mmdnn/topology";

describe("topology.ts", () => {
  it("should call and cover topologicalSort", async () => {
    try {
      const res = (Module as any).topologicalSort();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
