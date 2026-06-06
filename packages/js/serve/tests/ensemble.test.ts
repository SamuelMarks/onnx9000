import { describe, expect, it } from "vitest";
import { ModelEnsemble } from "../src/ensemble.js";

describe("ModelEnsemble", () => {
  it("should execute ensemble", async () => {
    const ens = new ModelEnsemble({
      name: "test",
      inputs: ["global_in"],
      outputs: { final: "n1.out" },
      nodes: [
        {
          id: "n1",
          type: "logic",
          inputs: { in: "global.global_in" },
          outputs: ["out"],
          logic: async (inputs) => inputs.in,
        },
      ],
    });

    const res = await ens.execute({ global_in: 42 }, {});
    expect(res.final).toBe(42);
  });
});
