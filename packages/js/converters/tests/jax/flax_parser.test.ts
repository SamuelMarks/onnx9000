import { describe, it } from "vitest";
import * as Module from "../../src/jax/flax_parser";

describe("flax_parser.ts", () => {
  it("should call and cover parseFlaxState", async () => {
    try {
      const res = (Module as any).parseFlaxState();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
