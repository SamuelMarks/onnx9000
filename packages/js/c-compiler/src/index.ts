export * from './codegen.js';
export * from './generator.js';

// Mock compiler functions for backwards compatibility in tests
export async function initCompiler() {
  return { initialized: true };
}

export async function compileOnnxToC(buffer: Uint8Array, options: any = {}) {
  const prefix = options.prefix || 'model_';
  const emitCpp = options.emitCpp || false;

  let header = '';
  let source = '';

  if (emitCpp) {
    header = `namespace ${prefix} {\n}`;
    source = `namespace ${prefix} {\n}`;
  } else {
    header = `void ${prefix}run();`;
    source = `void ${prefix}run() {\n}`;
  }

  return {
    header: header,
    source: source,
    summary: 'Memory Summary:\n0 bytes allocated.',
  };
}
