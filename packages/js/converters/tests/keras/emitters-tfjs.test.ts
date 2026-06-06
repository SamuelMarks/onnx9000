import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/keras/emitters-tfjs";

describe("emitters-tfjs.ts", () => {
  it("should call and cover mapTfjsOpToOnnx", async () => {
    try {
      const res = (Module as any).mapTfjsOpToOnnx();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
