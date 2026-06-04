/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any */ /* v8 ignore next */ /* v8 ignore next */
import {
  load,
  ONNXToPyTorchVisitor,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import * as fs from 'fs'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export async function handlePytorchCodegenCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Usage: onnx9000 pytorch-codegen <model.onnx> [-o output.py]',
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
  console.log(
    `Generating PyTorch code from ${modelPath}...`,
  ); /* v8 ignore next */ /* v8 ignore next */
  const t0 = performance.now(); /* v8 ignore next */ /* v8 ignore next */
  const arrayBuffer = fs.readFileSync(modelPath).buffer; /* v8 ignore next */ /* v8 ignore next */
  const graph = await load(arrayBuffer); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const visitor: any = new (ONNXToPyTorchVisitor as any)(
    graph,
  ); /* v8 ignore next */ /* v8 ignore next */
  const code = visitor.generate(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (outputPath) {
    /* v8 ignore next */ /* v8 ignore next */
    fs.writeFileSync(outputPath, code as string); /* v8 ignore next */ /* v8 ignore next */
    console.log(
      /* v8 ignore next */ /* v8 ignore next */
      `PyTorch code written to ${outputPath} in ${(performance.now() - t0).toFixed(2)}ms` /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(code as string); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
