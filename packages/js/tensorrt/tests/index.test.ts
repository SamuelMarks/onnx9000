import { describe, expect, it, vi } from "vitest";
import { Builder } from "../src/index.js";

vi.mock("../src/ffi.js", () => ({
  trtFfi: {
    lib: {
      createInferBuilder_INTERNAL: vi.fn().mockReturnValue({}),
      createNetworkV2: vi.fn().mockReturnValue({}),
      destroyInferBuilder: vi.fn(),
      destroyNetworkDefinition: vi.fn(),
      markOutput: vi.fn(),
    },
    getVersion: vi.fn().mockReturnValue([80, 6, 0]),
  },
}));

describe("TensorRT Builder", () => {
  it("should build", () => {
    const b = new Builder();
    expect(b.ptr).toBeDefined();

    const n = b.createNetwork();
    expect(n.ptr).toBeDefined();

    n.markOutput({ ptr: {} });

    n.destroy();
    b.destroy();
  });
});
