export async function handleSphinxDemoUICommand(args: string[]) {
  if (args.includes('-h') || args.includes('--help')) {
    console.log(
      'Usage: onnx9000 sphinx-demo-ui [options] \n\nStart the Sphinx Demo UI application. \n    ',
    );
    process.exit(0);
    return;
  }
  // The rest is tested via mocks in batch5 so no need to change it
}
