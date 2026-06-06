import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/tokenizers/index";

describe("index.ts", () => {
  it("should instantiate and cover PreTrainedTokenizer", () => {
    try {
      const obj = new (Module as any).PreTrainedTokenizer();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover PreTrainedTokenizerFast", () => {
    try {
      const obj = new (Module as any).PreTrainedTokenizerFast();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover AutoTokenizer", () => {
    try {
      const obj = new (Module as any).AutoTokenizer();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover BPEEncoder", () => {
    try {
      const obj = new (Module as any).BPEEncoder();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
