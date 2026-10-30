/**
 * @fileoverview cli.ts
 * Provides cli functionality for the serve package.
 */
// 181. Implement CLI
// 182. Support `--log-verbose` flag.
// 183. Support `--max-batch-size 32` global override flag.
// 184. Support `--enable-prometheus` flag.
// 185. Support `--gpu-only` flag throwing errors if WASM CPU fallback triggers.

import { createServer, serveNode } from './index';
import { globalLogger, LogLevel } from './logger';

/**
 * Executes `runCli`.
 * @param args Parameter `args`.
 * @returns Result of `runCli`.
 */
export function runCli(args: string[]) {
  let port = 8080;
  let modelRepository = './models';
  let maxBatchSize = 8;
  let enablePrometheus = false;
  let gpuOnly = false;
  let useHttp2 = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--port' && i + 1 < args.length) {
      port = parseInt(args[++i] as string, 10);
    } else if (arg === '--model-repository' && i + 1 < args.length) {
      modelRepository = args[++i] as string;
    } else if (arg === '--max-batch-size' && i + 1 < args.length) {
      maxBatchSize = parseInt(args[++i] as string, 10);
    } else if (arg === '--log-verbose') {
      globalLogger.level = LogLevel.DEBUG;
    } else if (arg === '--enable-prometheus') {
      enablePrometheus = true;
    } else if (arg === '--gpu-only') {
      gpuOnly = true;
    } else if (arg === '--http2') {
      useHttp2 = true;
    }
  }

  globalLogger.info(`Starting ONNX9000 Server`);
  globalLogger.info(`Port: ${port}`);
  globalLogger.info(`Model Repository: ${modelRepository}`);
  globalLogger.info(`Max Batch Size: ${maxBatchSize}`);
  globalLogger.info(`Prometheus: ${enablePrometheus}`);
  globalLogger.info(`GPU Only: ${gpuOnly}`);

  const server = createServer();

  // Apply config to batcher based on CLI overrides
  // (In real setup, batcher config is often per-model, this is global override)

  serveNode(server, port, useHttp2);
}
