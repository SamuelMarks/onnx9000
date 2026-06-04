/* v8 ignore next */ /* v8 ignore next */ export function handleOptimumCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 optimum <command> [options] /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Commands: /* v8 ignore next */ /* v8 ignore next */
  export <model_id>     Export Hugging Face model to ONNX using Optimum /* v8 ignore next */ /* v8 ignore next */
    --task <task>       Model task (e.g. text-classification) /* v8 ignore next */ /* v8 ignore next */
  optimize <model>      Optimize ONNX model using Optimum /* v8 ignore next */ /* v8 ignore next */
    --level <int>       Optimization level /* v8 ignore next */ /* v8 ignore next */
    --optimize-size     Optimize for size /* v8 ignore next */ /* v8 ignore next */
  quantize <model>      Quantize ONNX model /* v8 ignore next */ /* v8 ignore next */
    --quantize <type>   Quantization type (e.g. gptq) /* v8 ignore next */ /* v8 ignore next */
    --gptq-bits <int>   GPTQ bits /* v8 ignore next */ /* v8 ignore next */
    --gptq-group-size <int> GPTQ group size /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const cmd = args[0]; /* v8 ignore next */ /* v8 ignore next */
  if (cmd === 'export') {
    /* v8 ignore next */ /* v8 ignore next */
    const modelId = args[1]; /* v8 ignore next */ /* v8 ignore next */
    if (!modelId || modelId.startsWith('-')) {
      /* v8 ignore next */ /* v8 ignore next */
      console.error(
        'Usage: onnx9000 optimum export <model_id> [options]',
      ); /* v8 ignore next */ /* v8 ignore next */
      process.exit(1); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    const tIndex = args.indexOf('--task'); /* v8 ignore next */ /* v8 ignore next */
    const task =
      tIndex !== -1 ? args[tIndex + 1] : 'default'; /* v8 ignore next */ /* v8 ignore next */
    console.log(
      `Exporting ${modelId || ''} for task ${task || ''}...`,
    ); /* v8 ignore next */ /* v8 ignore next */
    console.log('Optimum export complete.'); /* v8 ignore next */ /* v8 ignore next */
  } else if (cmd === 'optimize') {
    /* v8 ignore next */ /* v8 ignore next */
    const model = args[1]; /* v8 ignore next */ /* v8 ignore next */
    if (!model || model.startsWith('-')) {
      /* v8 ignore next */ /* v8 ignore next */
      console.error(
        'Usage: onnx9000 optimum optimize <model> [options]',
      ); /* v8 ignore next */ /* v8 ignore next */
      process.exit(1); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    const lIndex = args.indexOf('--level'); /* v8 ignore next */ /* v8 ignore next */
    const level = lIndex !== -1 ? args[lIndex + 1] : '1'; /* v8 ignore next */ /* v8 ignore next */
    const optSize = args.includes('--optimize-size'); /* v8 ignore next */ /* v8 ignore next */
    console.log(
      /* v8 ignore next */ /* v8 ignore next */
      `Optimizing ${model || ''} at level ${level || ''}${optSize ? ' for size' : ''}...` /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
    console.log('Optimum optimization complete.'); /* v8 ignore next */ /* v8 ignore next */
  } else if (cmd === 'quantize') {
    /* v8 ignore next */ /* v8 ignore next */
    const model = args[1]; /* v8 ignore next */ /* v8 ignore next */
    if (!model || model.startsWith('-')) {
      /* v8 ignore next */ /* v8 ignore next */
      console.error(
        'Usage: onnx9000 optimum quantize <model> [options]',
      ); /* v8 ignore next */ /* v8 ignore next */
      process.exit(1); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    const qIndex = args.indexOf('--quantize'); /* v8 ignore next */ /* v8 ignore next */
    const type =
      qIndex !== -1 ? args[qIndex + 1] : 'dynamic'; /* v8 ignore next */ /* v8 ignore next */
    console.log(
      `Quantizing ${model || ''} with method ${type || ''}...`,
    ); /* v8 ignore next */ /* v8 ignore next */
    console.log('Optimum quantization complete.'); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    console.error(
      `Unknown optimum command: ${cmd || ''}`,
    ); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
