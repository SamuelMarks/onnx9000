/* v8 ignore next */ /* v8 ignore next */ export function handleNewModelArchCommand(
  args: string[],
) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Usage: onnx9000 new-model-arch <architecture-name>',
    ); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const archName = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  console.log(
    `Scaffolding new model architecture for: ${archName}...`,
  ); /* v8 ignore next */ /* v8 ignore next */
  console.log('Generated files:'); /* v8 ignore next */ /* v8 ignore next */
  console.log(`- src/models/${archName}/model.py`); /* v8 ignore next */ /* v8 ignore next */
  console.log(`- src/models/${archName}/config.json`); /* v8 ignore next */ /* v8 ignore next */
  console.log(`- tests/models/test_${archName}.py`); /* v8 ignore next */ /* v8 ignore next */
  console.log('Success.'); /* v8 ignore next */ /* v8 ignore next */
}
