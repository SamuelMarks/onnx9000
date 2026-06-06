import { describe, expect, it } from "vitest";
import * as tf from "../src/index.js";

describe("tfjs-shim", () => {
  it("should run ops", () => {
    expect(tf.version).toBeDefined();

    const a = tf.tensor([1, 2]);
    const b = tf.tensor([3, 4]);

    const c = tf.add(a, b);
    expect(c.shape).toBeDefined();
  });
});
