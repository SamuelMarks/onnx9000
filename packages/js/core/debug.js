/**
 * Debug script for ONNX parser
 */
export async function run() {
  const { load } = await import('./src/index.ts');
  try {
    const graph = await load(new Uint8Array([0x08, 0x01])); // Simple valid-ish proto header
    console.log('Success! Nodes:', graph.nodes.length);
  } catch (e) {
    console.error(e);
  }
}

/**
 * Main entry point for the debug script.
 * Checks environment variables and runs the script if requested.
 */
export async function main() {
  if (process.env.DEBUG_FORCE_RUN === 'true') {
    await run();
  }
}

main();
