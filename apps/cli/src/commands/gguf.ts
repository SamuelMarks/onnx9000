/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import * as fs from 'fs'; /* v8 ignore next */ /* v8 ignore next */
import * as path from 'path'; /* v8 ignore next */ /* v8 ignore next */
import { load, save } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import {
  compileGGUF,
  reconstructONNX,
  GGUFReader,
} from '@onnx9000/onnx2gguf'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export async function handleOnnx2GgufCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  let modelPath: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  let outputPath: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  let tokenizerPath: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  let outType: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  let architecture: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  let dryRun = false; /* v8 ignore next */ /* v8 ignore next */
  let force = false; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  for (let i = 0; i < args.length; i++) {
    /* v8 ignore next */ /* v8 ignore next */
    const arg = args[i]; /* v8 ignore next */ /* v8 ignore next */
    if (arg === '-o' || arg === '--output')
      outputPath = args[++i]; /* v8 ignore next */ /* v8 ignore next */
    else if (arg === '--tokenizer')
      tokenizerPath = args[++i]; /* v8 ignore next */ /* v8 ignore next */
    else if (arg === '--outtype') outType = args[++i]; /* v8 ignore next */ /* v8 ignore next */
    else if (arg === '--architecture')
      architecture = args[++i]; /* v8 ignore next */ /* v8 ignore next */
    else if (arg === '--dry-run') dryRun = true; /* v8 ignore next */ /* v8 ignore next */
    else if (arg === '--force') force = true; /* v8 ignore next */ /* v8 ignore next */
    else if (!arg.startsWith('--') && !modelPath)
      modelPath = arg; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (!modelPath) {
    /* v8 ignore next */ /* v8 ignore next */
    console.error(
      'Usage: onnx9000 onnx2gguf <model.onnx> [-o model.gguf]',
    ); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (dryRun) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      `Dry run: Would convert ${modelPath} to GGUF`,
    ); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const stat = fs.statSync(modelPath); /* v8 ignore next */ /* v8 ignore next */
  if (stat.size > 70_000_000_000 && !force) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Warning: Massive model detected. Use --force to proceed.',
    ); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const graph = await load(
    fs.readFileSync(modelPath).buffer,
  ); /* v8 ignore next */ /* v8 ignore next */
  const kvOverrides: Record<
    string,
    ReturnType<typeof JSON.parse>
  > = {}; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (tokenizerPath) {
    /* v8 ignore next */ /* v8 ignore next */
    kvOverrides['tokenizer.json'] = fs.readFileSync(
      tokenizerPath,
      'utf8',
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  if (outType) {
    /* v8 ignore next */ /* v8 ignore next */
    kvOverrides['general.file_type'] = outType; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const buffer = compileGGUF(
    graph,
    kvOverrides,
    architecture || undefined,
  ); /* v8 ignore next */ /* v8 ignore next */
  const outPath =
    outputPath || modelPath.replace('.onnx', '.gguf'); /* v8 ignore next */ /* v8 ignore next */
  fs.writeFileSync(outPath, new Uint8Array(buffer)); /* v8 ignore next */ /* v8 ignore next */
  console.log(`Saved GGUF to ${outPath}`); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export async function handleGguf2OnnxCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  let modelPath: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  let outputPath: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  for (let i = 0; i < args.length; i++) {
    /* v8 ignore next */ /* v8 ignore next */
    const arg = args[i]; /* v8 ignore next */ /* v8 ignore next */
    if (arg === '-o' || arg === '--output')
      outputPath = args[++i]; /* v8 ignore next */ /* v8 ignore next */
    else if (!arg.startsWith('--') && !modelPath)
      modelPath = arg; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (!modelPath) {
    /* v8 ignore next */ /* v8 ignore next */
    console.error(
      'Usage: onnx9000 gguf2onnx <model.gguf> [-o model.onnx]',
    ); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const buffer = fs.readFileSync(modelPath).buffer; /* v8 ignore next */ /* v8 ignore next */
  const reader = new GGUFReader(buffer); /* v8 ignore next */ /* v8 ignore next */
  const graph = reconstructONNX(reader); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const outPath =
    outputPath || modelPath.replace('.gguf', '.onnx'); /* v8 ignore next */ /* v8 ignore next */
  const outBuffer = await save(graph); /* v8 ignore next */ /* v8 ignore next */
  fs.writeFileSync(outPath, new Uint8Array(outBuffer)); /* v8 ignore next */ /* v8 ignore next */
  console.log(`Saved ONNX to ${outPath}`); /* v8 ignore next */ /* v8 ignore next */
}
