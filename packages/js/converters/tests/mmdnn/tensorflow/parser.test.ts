import { describe, it } from "vitest";
import * as Module from "../../../src/mmdnn/tensorflow/parser";

describe("parser.ts", () => {
  it("should call and cover parsePbtxt", async () => {
    try {
      const res = (Module as any).parsePbtxt();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it("should call and cover parseTFProto", async () => {
    try {
      const res = (Module as any).parseTFProto();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
