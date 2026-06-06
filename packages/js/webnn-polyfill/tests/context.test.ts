import { describe, expect, it } from "vitest";
import { PolyfillMLGraphBuilder } from "../src/builder.js";
import { PolyfillMLContext } from "../src/context.js";

describe("WebNN Context", () => {
  it("should compute", async () => {
    const ctx = new PolyfillMLContext();
    const limits = ctx.opSupportLimits();
    expect(limits.input).toBeDefined();

    const b = new PolyfillMLGraphBuilder(ctx);
    const i1 = b.input("a", { dataType: "float32", dimensions: [1] });
    const c1 = b.constant(
      { dataType: "float32", dimensions: [1] },
      new Float32Array([1]),
    );
    const out = b.add(i1, c1);

    const g = await b.build({ out });

    const res = await ctx.compute(
      g as any,
      { a: new Float32Array([2]) },
      { out: new Float32Array(1) },
    );
    expect(res).toBeDefined();
  });

  it("should handle tensor lifecycle", async () => {
    const ctx = new PolyfillMLContext();
    const t = await ctx.createTensor({ dataType: "float32", dimensions: [1] });
    const buf = new ArrayBuffer(4);
    await ctx.readTensor(t, buf);
    await ctx.writeTensor(t, buf);
    t.destroy();
  });
});
