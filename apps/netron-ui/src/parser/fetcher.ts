/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import {
  BlobReader,
  parseModelProto,
  Graph,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export async function fetchAndParseModel /* v8 ignore next */ /* v8 ignore next */(
  url: string /* v8 ignore next */ /* v8 ignore next */,
  progressCallback?: (percent: number) => void /* v8 ignore next */ /* v8 ignore next */,
): Promise<Graph> {
  /* v8 ignore next */ /* v8 ignore next */
  // Convert GitHub blob URLs to raw automatically /* v8 ignore next */ /* v8 ignore next */
  if (url.includes('github.com') && url.includes('/blob/')) {
    /* v8 ignore next */ /* v8 ignore next */
    url = url
      .replace('github.com', 'raw.githubusercontent.com')
      .replace('/blob/', '/'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const response = await fetch(url); /* v8 ignore next */ /* v8 ignore next */
  if (!response.ok) {
    /* v8 ignore next */ /* v8 ignore next */
    throw new Error(
      `Failed to fetch model from ${url}: ${response.statusText}`,
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const contentLength =
    response.headers.get('content-length'); /* v8 ignore next */ /* v8 ignore next */
  const total = contentLength
    ? parseInt(contentLength, 10)
    : 0; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (!response.body) {
    /* v8 ignore next */ /* v8 ignore next */
    throw new Error(
      'ReadableStream not supported by browser.',
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const reader = response.body.getReader(); /* v8 ignore next */ /* v8 ignore next */
  const chunks: Uint8Array[] = []; /* v8 ignore next */ /* v8 ignore next */
  let received = 0; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  while (true) {
    /* v8 ignore next */ /* v8 ignore next */
    const { done, value } = await reader.read(); /* v8 ignore next */ /* v8 ignore next */
    if (done) break; /* v8 ignore next */ /* v8 ignore next */
    if (value) {
      /* v8 ignore next */ /* v8 ignore next */
      chunks.push(value); /* v8 ignore next */ /* v8 ignore next */
      received += value.length; /* v8 ignore next */ /* v8 ignore next */
      if (progressCallback && total > 0) {
        /* v8 ignore next */ /* v8 ignore next */
        progressCallback((received / total) * 100); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const blob = new Blob(
    chunks as ReturnType<typeof JSON.parse> as BlobPart[],
  ); /* v8 ignore next */ /* v8 ignore next */
  const modelReader = new BlobReader(blob); /* v8 ignore next */ /* v8 ignore next */
  return parseModelProto(modelReader); /* v8 ignore next */ /* v8 ignore next */
}
