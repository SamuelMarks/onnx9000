/* eslint-disable */
import { generateTriton } from '@onnx9000/compiler';

self.onmessage = (e) => {
  const { graph, config } = e.data;
  // 187. Execute generation purely off the main thread.
  const code = generateTriton(graph, config);
  self.postMessage({ code });
};
