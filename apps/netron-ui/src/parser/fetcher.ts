/**
 * @fileoverview Provides utility functions for fetching and parsing ONNX models over the network.
 * Handles downloading binary data via Streams, transforming GitHub blobs to raw URLs,
 * and passing the result to the ONNX model parser.
 */
import { BlobReader, type Graph, parseModelProto } from "@onnx9000/core";

export async function fetchAndParseModel(
  url: string,
  progressCallback?: (percent: number) => void,
): Promise<Graph> {
  // Convert GitHub blob URLs to raw automatically
  if (url.includes("github.com") && url.includes("/blob/")) {
    url = url
      .replace("github.com", "raw.githubusercontent.com")
      .replace("/blob/", "/");
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch model from ${url}: ${response.statusText}`,
    );
  }

  const contentLength = response.headers.get("content-length");
  const total = contentLength ? parseInt(contentLength, 10) : 0;

  if (!response.body) {
    throw new Error("ReadableStream not supported by browser.");
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      received += value.length;
      if (progressCallback && total > 0) {
        progressCallback((received / total) * 100);
      }
    }
  }

  const blob = new Blob(chunks as ReturnType<typeof JSON.parse> as BlobPart[]);
  const modelReader = new BlobReader(blob);
  return parseModelProto(modelReader);
}
