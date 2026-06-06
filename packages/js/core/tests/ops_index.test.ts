import { describe, expect, it } from "vitest";
import { AddOp, MulOp, SubOp } from "../src/ops/index.js";

describe("ops", () => {
  it("should run operations", () => {
    const add = new AddOp();
    const res = add.execute(
      [
        { data: new Float32Array([1]) } as any,
        { data: new Float32Array([2]) } as any,
      ],
      {},
    );
    expect((res[0] as any).data[0]).toBe(3);

    const sub = new SubOp();
    const res2 = sub.execute(
      [
        { data: new Float32Array([1]) } as any,
        { data: new Float32Array([2]) } as any,
      ],
      {},
    );
    expect((res2[0] as any).data[0]).toBe(-1);

    const mul = new MulOp();
    const res3 = mul.execute(
      [
        { data: new Float32Array([1]) } as any,
        { data: new Float32Array([2]) } as any,
      ],
      {},
    );
    expect((res3[0] as any).data[0]).toBe(2);
  });
});
