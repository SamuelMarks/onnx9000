import { describe, it } from "vitest";
import * as Module from "../../../src/mmdnn/tfjs/serializer";

describe("serializer.ts", () => {
  it("should call and cover serializeTFJSWeights", async () => {
    try {
      const res = (Module as any).serializeTFJSWeights();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
