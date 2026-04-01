import { handleConvertCommand } from './commands/convert.js';
import { handleInspectCommand } from './commands/inspect.js';

async function main() {
  const args = process.argv.slice(2);
  if (args[0] === 'convert') {
    await handleConvertCommand(args.slice(1));
  } else if (args[0] === 'inspect') {
    await handleInspectCommand(args.slice(1));
  } else {
    console.error('Usage: onnx9000 <command> [options]');
    console.error('Available commands: convert, inspect');
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}
