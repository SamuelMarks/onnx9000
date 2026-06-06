import { describe, expect, it } from "vitest";
import { MatMul, Relu } from "../src/primitives.js";

describe("primitives", () => {
  it("should call relu", () => {
    const r = new Relu();
    const out = r.call({} as any);
    expect(out.name).toBe("Relu_out");
  });

  it("should call matmul", () => {
    const m = new MatMul();
    const out = m.call({} as any, {} as any);
    expect(out.name).toBe("MatMul_out");
  });
});
