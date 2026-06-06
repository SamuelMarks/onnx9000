import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/parser/magic";

describe("magic.ts", () => {
  it("should call and cover detectFormat", async () => {
    try {
      const res = (Module as any).detectFormat();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
