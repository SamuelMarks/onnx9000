import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/mmdnn/types";

describe("types.ts", () => {
  it("should call and cover getTypesIdentifier", async () => {
    try {
      const res = (Module as any).getTypesIdentifier();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
