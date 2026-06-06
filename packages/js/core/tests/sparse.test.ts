import { describe, it, expect } from "vitest";
import { denseToCoo, sparseToDense, getTypedArray } from "../src/sparse.js";
import { Tensor } from "../ir/tensor.js";

describe("sparse", () => {
  it("should convert to coo and back", () => {
    const t = new Tensor(
      "test",
      [4],
      "float32",
      false,
      false,
      new Float32Array([1, 0, 0, 2]),
    );
    const s = denseToCoo(t);
    expect(s.format).toBe("COO");

    const d = sparseToDense(s);
    expect(d.shape).toEqual([4]);
  });

  it("should get typed array", () => {
    expect(getTypedArray("float32", 10)).toBeInstanceOf(Float32Array);
  });
});
