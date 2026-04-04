// 181. Implement CLI
// 182. Support `--log-verbose` flag.
// 183. Support `--max-batch-size 32` global override flag.
// 184. Support `--enable-prometheus` flag.
// 185. Support `--gpu-only` flag throwing errors if WASM CPU fallback triggers.

import { createServer, serveNode } from './index';
import { globalLogger, LogLevel } from './logger';

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
      port = parseInt(args[++i] || '0', 10);
    } else if (arg === '--model-repository' && i + 1 < args.length) {
      modelRepository = args[++i] || '';
    } else if (arg === '--max-batch-size' && i + 1 < args.length) {
      maxBatchSize = parseInt(args[++i] || '0', 10);
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

// If invoked directly
if (typeof require !== 'undefined' && require.main === module) {
  /* v8 ignore start */
  runCli(process.argv.slice(2));
}
/* v8 ignore stop */
