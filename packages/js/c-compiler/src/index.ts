import { BufferReader, parseModelProto, Graph } from '@onnx9000/core';
import { CGenerator } from './generator.js';

/**
 * Options configuring the compilation process from ONNX to C/C++.
 */
export interface CompileOptions {
  /** Optional prefix for generated symbols (default: 'model_'). */
  prefix?: string;
  /** Whether to emit C++ standard constructs like std::vector (default: false). */
  emitCpp?: boolean;
  /** The target architecture or board type. */
  target?: string;
  /** Disables the standard <math.h> include. */
  noMathH?: boolean;
  /** Disables optimization passes during code generation. */
  noOpt?: boolean;
  /** Ensures memory buffers are aligned to the specified byte boundary. */
  align?: number;
  /** The number of spaces to use for indentation in emitted code (default: 4). */
  indentSpaces?: number;
}

/**
 * The resulting components of the C/C++ compilation.
 */
export interface CompileResult {
  /** The generated C/C++ header (.h / .hpp) code. */
  header: string;
  /** The generated C/C++ source (.c / .cpp) code. */
  source: string;
  /** A summary comment listing memory usage and footprint bounds. */
  summary: string;
}

export async function initCompiler() {
  // Backwards compatibility if needed, but no longer requires Pyodide
  return { initialized: true };
}

/**
 * Compile an ONNX model to C/C++.
 *
 * @param onnxBuffer The ONNX model binary buffer.
 * @param options Compilation options configuring code generation behavior.
 * @returns An object containing the generated header, source, and memory summary.
 */
export async function compileOnnxToC(
  onnxBuffer: Uint8Array,
  options: CompileOptions = {},
): Promise<CompileResult> {
  const prefix = options.prefix ?? 'model_';
  const emitCpp = options.emitCpp ?? false;

  const reader = new BufferReader(onnxBuffer);
  const graph = await parseModelProto(reader);

  const generator = new CGenerator(graph, prefix, emitCpp);

  return {
    header: generator.generateHeader(),
    source: generator.generateSource(),
    summary: generator.generateSummary(),
  };
}
