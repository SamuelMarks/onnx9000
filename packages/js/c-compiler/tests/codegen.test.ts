import { describe, expect, it } from "vitest";
import { CFamilyCodegen, PythonFamilyCodegen } from "../src/codegen.js";

describe("codegen", () => {
  it("should codegen C", () => {
    const cg = new CFamilyCodegen();
    const graph: any = { name: "test", nodes: [{ opType: "Add" }] };
    const code = cg.visit(graph);
    expect(code).toContain("void forward_test()");
    expect(code).toContain("op_add");
  });

  it("should codegen Python", () => {
    const cg = new PythonFamilyCodegen();
    const graph: any = { name: "test", nodes: [{ opType: "Add" }] };
    const code = cg.visit(graph);
    expect(code).toContain("def forward_test");
    expect(code).toContain("add()");
  });
});
