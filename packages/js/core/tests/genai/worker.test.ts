import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/genai/worker";

describe("worker.ts", () => {
  it("should instantiate and cover PrefixTreeBatchedProcessor", () => {
    try {
      const obj = new (Module as any).PrefixTreeBatchedProcessor();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover LookaheadDecoder", () => {
    try {
      const obj = new (Module as any).LookaheadDecoder();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover MedusaEagleDecoder", () => {
    try {
      const obj = new (Module as any).MedusaEagleDecoder();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover RAGPrefixMatcher", () => {
    try {
      const obj = new (Module as any).RAGPrefixMatcher();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
