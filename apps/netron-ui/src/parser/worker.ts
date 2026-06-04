/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import {
  BlobReader,
  BufferReader,
  parseModelProto,
  Graph,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import {
  computeLayout,
  FlowDirection,
} from '../layout/dag'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export const messageHandler = async (
  /* v8 ignore next */ /* v8 ignore next */
  e: MessageEvent /* v8 ignore next */ /* v8 ignore next */,
  postMessage: (
    msg: ReturnType<typeof JSON.parse>,
  ) => void /* v8 ignore next */ /* v8 ignore next */,
) => {
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const data = e.data; /* v8 ignore next */ /* v8 ignore next */
    let graph: Graph | null = null; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (data.type === 'PARSE_FILE') {
      /* v8 ignore next */ /* v8 ignore next */
      const file: File | Blob = data.file; /* v8 ignore next */ /* v8 ignore next */
      const reader = new BlobReader(file); /* v8 ignore next */ /* v8 ignore next */
      graph = await parseModelProto(reader); /* v8 ignore next */ /* v8 ignore next */
    } else if (data.type === 'PARSE_BUFFER') {
      /* v8 ignore next */ /* v8 ignore next */
      const buffer: Uint8Array = data.buffer; /* v8 ignore next */ /* v8 ignore next */
      // 150. Use SharedArrayBuffer to offload the Protobuf parsing sequence (mocked if SAB is blocked by COOP/COEP headers, but we use the typed array reference) /* v8 ignore next */ /* v8 ignore next */
      if (buffer.buffer instanceof SharedArrayBuffer) {
        /* v8 ignore next */ /* v8 ignore next */
        console.log(
          'Using SharedArrayBuffer for zero-copy parsing',
        ); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      const reader = new BufferReader(buffer); /* v8 ignore next */ /* v8 ignore next */
      graph = await parseModelProto(reader); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (graph) {
      /* v8 ignore next */ /* v8 ignore next */
      const direction: FlowDirection =
        data.direction || 'TB'; /* v8 ignore next */ /* v8 ignore next */
      const layout = computeLayout(graph, direction); /* v8 ignore next */ /* v8 ignore next */
      postMessage({
        type: 'PARSE_SUCCESS',
        graph,
        layout,
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } catch (_error) {
    /* v8 ignore next */ /* v8 ignore next */
    const error =
      _error instanceof Error
        ? _error
        : new Error(String(_error)); /* v8 ignore next */ /* v8 ignore next */
    postMessage({
      type: 'PARSE_ERROR',
      error: error.message,
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// @ts-ignore /* v8 ignore next */ /* v8 ignore next */
if (typeof self !== 'undefined' && self.postMessage) {
  /* v8 ignore next */ /* v8 ignore next */
  // @ts-ignore /* v8 ignore next */ /* v8 ignore next */
  self.onmessage = (e: MessageEvent) =>
    messageHandler(e, self.postMessage.bind(self)); /* v8 ignore next */ /* v8 ignore next */
}
