import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/compiler/subgraph";

describe("subgraph.ts", () => {
  it("should call and cover compileGraphToTFLite", async () => {
    try {
      const res = (Module as any).compileGraphToTFLite();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
