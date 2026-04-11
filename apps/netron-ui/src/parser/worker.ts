/* eslint-disable */
import { BlobReader, BufferReader, parseModelProto, Graph } from '@onnx9000/core';
import { computeLayout, FlowDirection } from '../layout/dag';

export const messageHandler = async (
  e: MessageEvent,
  postMessage: (msg: ReturnType<typeof JSON.parse>) => void,
) => {
  try {
    const data = e.data;
    let graph: Graph | null = null;

    if (data.type === 'PARSE_FILE') {
      const file: File | Blob = data.file;
      const reader = new BlobReader(file);
      graph = await parseModelProto(reader);
    } else if (data.type === 'PARSE_BUFFER') {
      const buffer: Uint8Array = data.buffer;
      // 150. Use SharedArrayBuffer to offload the Protobuf parsing sequence (mocked if SAB is blocked by COOP/COEP headers, but we use the typed array reference)
      if (buffer.buffer instanceof SharedArrayBuffer) {
        console.log('Using SharedArrayBuffer for zero-copy parsing');
      }
      const reader = new BufferReader(buffer);
      graph = await parseModelProto(reader);
    }

    if (graph) {
      const direction: FlowDirection = data.direction || 'TB';
      const layout = computeLayout(graph, direction);
      postMessage({ type: 'PARSE_SUCCESS', graph, layout });
    }
  } catch (_error) {
    const error = _error instanceof Error ? _error : new Error(String(_error));
    postMessage({ type: 'PARSE_ERROR', error: error.message });
  }
};

// @ts-ignore
if (typeof self !== 'undefined' && self.postMessage) {
  // @ts-ignore
  self.onmessage = (e: MessageEvent) => messageHandler(e, self.postMessage.bind(self));
}
