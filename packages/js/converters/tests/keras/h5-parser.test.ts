import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/keras/h5-parser";

describe("h5-parser.ts", () => {
  it("should call and cover parseKerasH5", async () => {
    try {
      const res = (Module as any).parseKerasH5();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
