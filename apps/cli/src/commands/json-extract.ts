import { load } from '@onnx9000/core';
import * as fs from 'fs';

export async function handleJsonExtractCommand(args: string[]) {
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    console.log('Usage: onnx9000 json-extract <model.onnx> [-o output.json]');
    process.exit(0);
    return;
  }

  const modelPath = args[0] || '';
  let outputPath = '';
  if (args[1] === '-o' || args[1] === '--output') {
    outputPath = args[2] || '';
  }

  console.log(`Extracting JSON from ${modelPath}...`);
  const t0 = performance.now();
  const arrayBuffer = fs.readFileSync(modelPath).buffer;
  const graph = await load(arrayBuffer);

  const jsonString = JSON.stringify(
    graph,
    (key, value) => {
      if (key === 'data' && ArrayBuffer.isView(value)) {
        return `[Buffer: ${value.byteLength.toString()} bytes]`;
      }
      if (typeof value === 'bigint') {
        return value.toString() + 'n';
      }
      return value as unknown;
    },
    2,
  );

  if (outputPath) {
    fs.writeFileSync(outputPath, jsonString);
    console.log(
      `Extracted JSON written to ${outputPath} in ${(performance.now() - t0).toFixed(2)}ms`,
    );
  } else {
    console.log(jsonString);
  }
}
