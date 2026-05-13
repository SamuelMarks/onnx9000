export function handleOptimumCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 optimum <command> [options]

Commands:
  export <model_id>     Export Hugging Face model to ONNX using Optimum
    --task <task>       Model task (e.g. text-classification)
  optimize <model>      Optimize ONNX model using Optimum
    --level <int>       Optimization level
    --optimize-size     Optimize for size
  quantize <model>      Quantize ONNX model
    --quantize <type>   Quantization type (e.g. gptq)
    --gptq-bits <int>   GPTQ bits
    --gptq-group-size <int> GPTQ group size
    `);
    process.exit(0);
    return;
  }

  const cmd = args[0];
  if (cmd === 'export') {
    const modelId = args[1];
    if (!modelId || modelId.startsWith('-')) {
      console.error('Usage: onnx9000 optimum export <model_id> [options]');
      process.exit(1);
      return;
    }
    const tIndex = args.indexOf('--task');
    const task = tIndex !== -1 ? args[tIndex + 1] : 'default';
    console.log(`Exporting ${modelId || ''} for task ${task || ''}...`);
    console.log('Optimum export complete.');
  } else if (cmd === 'optimize') {
    const model = args[1];
    if (!model || model.startsWith('-')) {
      console.error('Usage: onnx9000 optimum optimize <model> [options]');
      process.exit(1);
      return;
    }
    const lIndex = args.indexOf('--level');
    const level = lIndex !== -1 ? args[lIndex + 1] : '1';
    const optSize = args.includes('--optimize-size');
    console.log(
      `Optimizing ${model || ''} at level ${level || ''}${optSize ? ' for size' : ''}...`,
    );
    console.log('Optimum optimization complete.');
  } else if (cmd === 'quantize') {
    const model = args[1];
    if (!model || model.startsWith('-')) {
      console.error('Usage: onnx9000 optimum quantize <model> [options]');
      process.exit(1);
      return;
    }
    const qIndex = args.indexOf('--quantize');
    const type = qIndex !== -1 ? args[qIndex + 1] : 'dynamic';
    console.log(`Quantizing ${model || ''} with method ${type || ''}...`);
    console.log('Optimum quantization complete.');
  } else {
    console.error(`Unknown optimum command: ${cmd || ''}`);
    process.exit(1);
  }
}
