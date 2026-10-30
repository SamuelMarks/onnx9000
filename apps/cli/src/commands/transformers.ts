/**
 * @fileoverview transformers module for cli.
 */
import { pipeline } from '@onnx9000/transformers';

/**
 * Executes `handleTransformersCommand`.
 * @param args Parameter `args`.
 * @returns Result of `handleTransformersCommand`.
 */
export async function handleTransformersCommand(args: string[]) {
  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    console.log('Usage: onnx9000 transformers <task> [input_string]');
    console.log('Supported tasks include: text-classification, text-generation, ...');
    return;
  }

  const task = args[0];
  const inputString = args.slice(1).join(' ') || 'I love ONNX9000!';

  console.log(`Initializing Transformers.js Pipeline for task: ${task}...`);
  try {
    const pipe = await pipeline(task);
    console.log(`Running inference on: "${inputString}"...`);
    const result = await pipe(inputString);
    console.log('Result:', JSON.stringify(result, null, 2));
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error('Pipeline execution failed:', msg);
  }
}
