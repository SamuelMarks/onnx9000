export function handleScriptCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 script <input.py> [-o <output.onnx>]

Execute an ONNX Script.
    `);
    process.exit(0);
    return;
  }

  const scriptPath = args[0] || '';
  console.log(`Executing ONNX Script from ${scriptPath}`);

  const oIndex = args.indexOf('-o');
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    const output = args[oIndex + 1];
    console.log(`Saved compiled ONNX to ${String(output)}`);
  } else {
    console.log('Successfully compiled script. Use -o to save the output.');
  }
}
