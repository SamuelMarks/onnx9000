/**
 * @fileoverview worker.ts
 * Provides worker functionality for the triton-compiler package.
 */
import { generateTriton } from '@onnx9000/compiler';

self.onmessage = (e) => {
  const { graph, config } = e.data;
  // 187. Execute generation purely off the main thread.
  const code = generateTriton(graph, config);
  self.postMessage({ code });
};
