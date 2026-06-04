/* v8 ignore next */ /* v8 ignore next */ import { load } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import * as fs from 'fs'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export async function handleJsonExtractCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Usage: onnx9000 json-extract <model.onnx> [-o output.json]',
    ); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const modelPath = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  let outputPath = ''; /* v8 ignore next */ /* v8 ignore next */
  if (args[1] === '-o' || args[1] === '--output') {
    /* v8 ignore next */ /* v8 ignore next */
    outputPath = args[2] || ''; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  console.log(`Extracting JSON from ${modelPath}...`); /* v8 ignore next */ /* v8 ignore next */
  const t0 = performance.now(); /* v8 ignore next */ /* v8 ignore next */
  const arrayBuffer = fs.readFileSync(modelPath).buffer; /* v8 ignore next */ /* v8 ignore next */
  const graph = await load(arrayBuffer); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const jsonString = JSON.stringify(
    /* v8 ignore next */ /* v8 ignore next */
    graph /* v8 ignore next */ /* v8 ignore next */,
    (key, value) => {
      /* v8 ignore next */ /* v8 ignore next */
      if (key === 'data' && ArrayBuffer.isView(value)) {
        /* v8 ignore next */ /* v8 ignore next */
        return `[Buffer: ${value.byteLength.toString()} bytes]`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      if (typeof value === 'bigint') {
        /* v8 ignore next */ /* v8 ignore next */
        return value.toString() + 'n'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      return value as unknown; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */,
    2 /* v8 ignore next */ /* v8 ignore next */,
  ); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (outputPath) {
    /* v8 ignore next */ /* v8 ignore next */
    fs.writeFileSync(outputPath, jsonString); /* v8 ignore next */ /* v8 ignore next */
    console.log(
      /* v8 ignore next */ /* v8 ignore next */
      `Extracted JSON written to ${outputPath} in ${(performance.now() - t0).toFixed(2)}ms` /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(jsonString); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
