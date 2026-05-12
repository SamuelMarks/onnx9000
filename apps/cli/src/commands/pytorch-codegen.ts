/* eslint-disable @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any */
import { load, ONNXToPyTorchVisitor } from '@onnx9000/core';
import * as fs from 'fs';

export async function handlePytorchCodegenCommand(args: string[]) {
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    console.log('Usage: onnx9000 pytorch-codegen <model.onnx> [-o output.py]');
    process.exit(0);
    return;
  }

  const modelPath = args[0] || '';
  let outputPath = '';
  if (args[1] === '-o' || args[1] === '--output') {
    outputPath = args[2] || '';
  }

  console.log(`Generating PyTorch code from ${modelPath}...`);
  const t0 = performance.now();
  const arrayBuffer = fs.readFileSync(modelPath).buffer;
  const graph = await load(arrayBuffer);

  const visitor: any = new (ONNXToPyTorchVisitor as any)(graph);
  const code = visitor.generate();

  if (outputPath) {
    fs.writeFileSync(outputPath, code as string);
    console.log(
      `PyTorch code written to ${outputPath} in ${(performance.now() - t0).toFixed(2)}ms`,
    );
  } else {
    console.log(code as string);
  }
}
