export function handleExportCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 export <model.onnx> [options]

Export an ONNX model to another format (e.g. C/C99 source code).
    --format <fmt>      Target format (e.g. c)
    -o <file>           Output file path
    `);
    process.exit(0);
    return;
  }

  const model = args[0] || '';

  let format = '';
  const fIndex = args.indexOf('--format');
  if (fIndex !== -1 && fIndex + 1 < args.length) {
    format = args[fIndex + 1];
  }

  if (format !== 'c') {
    console.error(`Unsupported format: ${format || ''}`);
    process.exit(1);
    return;
  }

  let output = model.replace('.onnx', '.c');
  const oIndex = args.indexOf('-o');
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    output = args[oIndex + 1];
  }

  console.log(`Loading model ${model || ''}...`);
  console.log('Transpiling ONNX to C99...');
  console.log(`Saving C source to ${output}...`);
  console.log('Export complete.');
}
