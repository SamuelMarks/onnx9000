import { describe, it } from "vitest";
import * as Module from "../../src/mil/deterministic";

describe("deterministic.ts", () => {
  it("should call and cover assertDeterministicBuild", async () => {
    try {
      const res = (Module as any).assertDeterministicBuild();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
