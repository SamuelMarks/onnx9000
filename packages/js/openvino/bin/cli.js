#!/usr/bin/env node

import { load } from '@onnx9000/core';
import { OpenVinoExporter } from '../dist/index.js';
import fs from 'fs';
import path from 'path';

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log('Usage: onnx9000-openvino <model.onnx> [options]');
    console.log('Options:');
    console.log('  -o, --output_dir <dir>  Output directory (default: .)');
    console.log('  --fp16                  Compress weights to FP16');
    console.log('  --dynamic-batch         Set batch size to dynamic (-1)');
    console.log('  --shape <shape>         Override input shape (e.g. input:[1,3,224,224])');
    process.exit(0);
  }

  const modelPath = args[0];
  let outputDir = '.';
  let fp16 = false;
  let dynamicBatch = false;
  let shapeOverride = null;

  for (let i = 1; i < args.length; i++) {
    if ((args[i] === '-o' || args[i] === '--output_dir') && i + 1 < args.length) {
      outputDir = args[i + 1];
      i++;
    } else if (args[i] === '--fp16') {
      fp16 = true;
    } else if (args[i] === '--dynamic-batch') {
      dynamicBatch = true;
    } else if (args[i] === '--shape' && i + 1 < args.length) {
      shapeOverride = args[i + 1];
      i++;
    }
  }

  console.log(`[1/3] Loading ONNX model: ${modelPath}`);
  const buffer = fs.readFileSync(modelPath);
  const graph = load(new Uint8Array(buffer));

  if (dynamicBatch) {
    for (const input of graph.inputs) {
      if (input.shape.length > 0) {
        input.shape[0] = -1;
      }
    }
  }

  if (shapeOverride) {
    const parts = shapeOverride.split(':');
    if (parts.length === 2) {
      const name = parts[0];
      const shapeStr = parts[1].replace(/[\[\]]/g, '');
      const shape = shapeStr.split(',').map((s) => parseInt(s, 10));
      const input = graph.inputs.find((i) => i.name === name);
      if (input) {
        input.shape = shape;
      }
    }
  }

  console.log(`[2/3] Compiling to OpenVINO IR (XML & BIN)...`);
  const exporter = new OpenVinoExporter(graph, { compressToFp16: fp16 });
  const { xml, bin } = exporter.export();

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const baseName = path.basename(modelPath, '.onnx');
  const xmlPath = path.join(outputDir, `${baseName}.xml`);
  const binPath = path.join(outputDir, `${baseName}.bin`);
  const mappingPath = path.join(outputDir, `${baseName}.mapping`);

  console.log(`[3/3] Writing outputs to ${outputDir}/`);
  fs.writeFileSync(xmlPath, xml);
  fs.writeFileSync(binPath, bin);
  fs.writeFileSync(mappingPath, '<?xml version="1.0" ?>\n<mapping>\n</mapping>');

  console.log(`Successfully exported OpenVINO model.`);
}

/* v8 ignore start */
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
/* v8 ignore stop */

export { main };
