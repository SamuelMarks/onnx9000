import { describe, expect, it } from "vitest";
import { SparseTensor, Tensor } from "../src/ir/tensor.js";

describe("Tensor", () => {
  it("should create and format", () => {
    const t = new Tensor("t1", [2, 2], "float32");
    expect(t.size).toBe(4);

    t.data = new Float32Array([1, 2, 3, 4]);
    expect(t.formatData()).toContain("1, 2, 3, 4");

    const t2 = t.copy();
    expect(t2.name).toBe("t1");
  });

  it("should create sparse", () => {
    const t = new SparseTensor("s", [10], "COO");
    expect(t.format).toBe("COO");
  });
});
