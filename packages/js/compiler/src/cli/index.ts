#!/usr/bin/env node

import * as fs from 'fs';
import * as path from 'path';

function printHelp() {
  console.log(`
Usage:
  npx @onnx9000/compiler compile <model.onnx> [options]

Options:
  --target-backend=<name>    Target backend (e.g., @onnx9000/backend-web)
  --optimize-level=<level>   Optimization level (O0, O1, O2, O3)
  --output=<path>            Output path for the compiled payload
  --help                     Print this help message
`);
}

function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args[0] !== 'compile') {
    printHelp();
    process.exit(args.includes('--help') ? 0 : 1);
  }

  const modelPath = args[1];
  if (!modelPath || modelPath.startsWith('--')) {
    console.error('Error: You must provide a path to an ONNX model.');
    process.exit(1);
  }

  const options: Record<string, string> = {};
  for (let i = 2; i < args.length; i++) {
    const arg = args[i];
    if (arg && arg.startsWith('--')) {
      const parts = arg.split('=');
      const key = parts[0];
      const value = parts[1];
      if (key) {
        options[key.substring(2)] = value || 'true';
      }
    }
  }

  if (!fs.existsSync(modelPath)) {
    console.error(`Error: Model file not found at ${modelPath}`);
    process.exit(1);
  }

  const backend = options['target-backend'] || '@onnx9000/backend-web';
  const optLevel = options['optimize-level'] || 'O3';
  const outPath = options['output'] || modelPath.replace('.onnx', '.bin');

  console.log(`Compiling ${modelPath} for ${backend} at level ${optLevel}...`);

  // Mock compilation logic that creates a dummy bin file representing the compiled inference payload.
  // In a real implementation, this would parse the ONNX, invoke the target backend's AOT compiler,
  // and output WASM/WGSL/CoreML binaries.
  const payload = JSON.stringify({
    compiler: '@onnx9000/compiler',
    version: '1.0.0',
    backend,
    optLevel,
    originalModel: path.basename(modelPath),
    timestamp: new Date().toISOString(),
  });

  fs.writeFileSync(outPath, payload);
  console.log(`Successfully generated compiled inference payload at ${outPath}`);
  console.log(`Size: ${String(Buffer.byteLength(payload))} bytes`);
}

if (import.meta.url === `file://${process.argv[1] || ''}`) {
  try {
    main();
  } catch (err: unknown) {
    console.error(err);
    process.exit(1);
  }
}

// Export for testing
export { main };
