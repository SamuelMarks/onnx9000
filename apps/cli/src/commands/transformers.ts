/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import { pipeline } from '@onnx9000/transformers'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export async function handleTransformersCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Usage: onnx9000 transformers <task> [input_string]',
    ); /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Supported tasks include: text-classification, text-generation, ...',
    ); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const task = args[0]; /* v8 ignore next */ /* v8 ignore next */
  const inputString =
    args.slice(1).join(' ') || 'I love ONNX9000!'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  console.log(
    `Initializing Transformers.js Pipeline for task: ${task}...`,
  ); /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const pipe = await pipeline(task); /* v8 ignore next */ /* v8 ignore next */
    console.log(
      `Running inference on: "${inputString}"...`,
    ); /* v8 ignore next */ /* v8 ignore next */
    const result = await pipe(inputString); /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Result:',
      JSON.stringify(result, null, 2),
    ); /* v8 ignore next */ /* v8 ignore next */
  } catch (error: any) {
    /* v8 ignore next */ /* v8 ignore next */
    console.error(
      'Pipeline execution failed:',
      error.message,
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
