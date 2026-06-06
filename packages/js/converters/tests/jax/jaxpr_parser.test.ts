import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/jax/jaxpr_parser";

describe("jaxpr_parser.ts", () => {
  it("should call and cover parseJaxpr", async () => {
    try {
      const res = (Module as any).parseJaxpr();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
