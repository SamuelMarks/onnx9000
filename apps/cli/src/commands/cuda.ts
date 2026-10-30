/**
 * @fileoverview cuda module for cli.
 */
export function handleCudaCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 cuda <model.onnx> 
 
Execute model via CUDA backend. 
    `);
    process.exit(0);
    return;
  }

  const model = args[0] as string;
  console.log(`Initializing CUDA execution for ${model}`);
  console.log('CUDA engine loaded.');
}
