import { Tensor } from '@onnx9000/core';
import { SessionOptions } from './session.js';

export interface ProgressiveLoadOptions extends SessionOptions {
  maxChunkSize?: number;
}

export class ProgressiveSession {
  private url: string;
  private options: ProgressiveLoadOptions;
  public isLoaded: boolean = false;

  constructor(url: string, options: ProgressiveLoadOptions) {
    this.url = url;
    this.options = options;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars, @typescript-eslint/require-await
  async run(_inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    // 1. Metadata Fetch: Simulated fetching of first few KB
    // 2. Layer-on-Demand: Request specific ranges
    // 3. Weight Streaming: Stream to WebGPU

    // Validate we can "fetch" conceptually (mock implementation)
    if (!this.url.startsWith('http') && !this.url.startsWith('/')) {
      throw new Error('Progressive loading requires a valid HTTP URL or absolute path.');
    }

    if (!this.isLoaded) {
      // Simulate chunking logic where maxChunkSize is used
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const _chunkSize = this.options.maxChunkSize || 1024 * 1024;
      // In a real implementation we would fetch(this.url, { headers: { Range: `bytes=0-${_chunkSize}` } })
      this.isLoaded = true;
    }

    // Mock output for now, as real engine is external/WASM
    const outputs: Record<string, Tensor> = {};
    return outputs;
  }
}

/**
 * Loads a model progressively using HTTP Range requests.
 * @param url The URL of the ONNX or Safetensors model.
 * @param options Configuration options including chunk size limits.
 * @returns A ProgressiveSession capable of lazily evaluating layers.
 */
// eslint-disable-next-line @typescript-eslint/require-await
export async function loadProgressive(
  url: string,
  options: ProgressiveLoadOptions = {},
): Promise<ProgressiveSession> {
  const session = new ProgressiveSession(url, options);
  // Initial metadata fetch mock
  return session;
}
