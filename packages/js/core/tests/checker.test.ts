import { describe, it, expect } from "vitest";
import {
  check_model,
  check_model_async,
  ValidationContext,
  SchemaRegistry,
} from "../src/checker.js";

describe("checker", () => {
  it("should validate valid model", () => {
    const model: any = {
      ir_version: 8,
      producer_name: "test",
      opset_import: [{ domain: "ai.onnx", version: 15 }],
      graph: {
        inputs: [{ name: "in", data_type: "float32", shape: [1] }],
        outputs: ["out"],
        initializers: [],
        nodes: [
          { op_type: "Relu", inputs: ["in"], outputs: ["out"], attributes: {} },
        ],
      },
    };

    expect(check_model(model)).toBe(true);
  });

  it("should throw on invalid model", () => {
    const model: any = {
      ir_version: 1, // invalid
      graph: {},
    };

    expect(() => check_model(model)).toThrow();
  });

  it("should validate async", async () => {
    const model: any = {
      ir_version: 8,
      producer_name: "test",
      opset_import: [{ domain: "ai.onnx", version: 15 }],
      graph: {
        inputs: [{ name: "in", data_type: "float32", shape: [1] }],
        outputs: ["out"],
        initializers: [],
        nodes: [
          { op_type: "Relu", inputs: ["in"], outputs: ["out"], attributes: {} },
        ],
      },
    };
    expect(await check_model_async(model)).toBe(true);
  });

  it("should manage schema registry", () => {
    const reg = new SchemaRegistry();
    expect(reg.get_schema("Conv", 15)).toBeDefined();
    expect(() => reg.get_schema("Unknown", 15)).toThrow();

    reg.register_custom_schema("custom", 1, { MyOp: { attr: "string" } });
    expect(reg.get_schema("MyOp", 1, "custom")).toBeDefined();
  });
});
