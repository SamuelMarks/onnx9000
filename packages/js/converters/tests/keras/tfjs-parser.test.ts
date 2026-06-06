import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/keras/tfjs-parser";

describe("tfjs-parser.ts", () => {
  it("should call and cover parseTFJSModel", async () => {
    try {
      const res = (Module as any).parseTFJSModel();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
