import * as fs from 'fs';
import * as path from 'path';
import { load, save } from '@onnx9000/core';
import { compileGGUF, reconstructONNX, GGUFReader } from '@onnx9000/onnx2gguf';

export async function handleOnnx2GgufCommand(args: string[]) {
  let modelPath: string | null = null;
  let outputPath: string | null = null;
  let tokenizerPath: string | null = null;
  let outType: string | null = null;
  let architecture: string | null = null;
  let dryRun = false;
  let force = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '-o' || arg === '--output') outputPath = args[++i];
    else if (arg === '--tokenizer') tokenizerPath = args[++i];
    else if (arg === '--outtype') outType = args[++i];
    else if (arg === '--architecture') architecture = args[++i];
    else if (arg === '--dry-run') dryRun = true;
    else if (arg === '--force') force = true;
    else if (!arg.startsWith('--') && !modelPath) modelPath = arg;
  }

  if (!modelPath) {
    console.error('Usage: onnx9000 onnx2gguf <model.onnx> [-o model.gguf]');
    process.exit(1);
  }

  if (dryRun) {
    console.log(`Dry run: Would convert ${modelPath} to GGUF`);
    return;
  }

  const stat = fs.statSync(modelPath);
  if (stat.size > 70_000_000_000 && !force) {
    console.log('Warning: Massive model detected. Use --force to proceed.');
    return;
  }

  const graph = await load(fs.readFileSync(modelPath).buffer);
  const kvOverrides: Record<string, ReturnType<typeof JSON.parse>> = {};

  if (tokenizerPath) {
    kvOverrides['tokenizer.json'] = fs.readFileSync(tokenizerPath, 'utf8');
  }
  if (outType) {
    kvOverrides['general.file_type'] = outType;
  }

  const buffer = compileGGUF(graph, kvOverrides, architecture || undefined);
  const outPath = outputPath || modelPath.replace('.onnx', '.gguf');
  fs.writeFileSync(outPath, new Uint8Array(buffer));
  console.log(`Saved GGUF to ${outPath}`);
}

export async function handleGguf2OnnxCommand(args: string[]) {
  let modelPath: string | null = null;
  let outputPath: string | null = null;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '-o' || arg === '--output') outputPath = args[++i];
    else if (!arg.startsWith('--') && !modelPath) modelPath = arg;
  }

  if (!modelPath) {
    console.error('Usage: onnx9000 gguf2onnx <model.gguf> [-o model.onnx]');
    process.exit(1);
  }

  const buffer = fs.readFileSync(modelPath).buffer;
  const reader = new GGUFReader(buffer);
  const graph = reconstructONNX(reader);

  const outPath = outputPath || modelPath.replace('.gguf', '.onnx');
  const outBuffer = await save(graph);
  fs.writeFileSync(outPath, new Uint8Array(outBuffer));
  console.log(`Saved ONNX to ${outPath}`);
}
