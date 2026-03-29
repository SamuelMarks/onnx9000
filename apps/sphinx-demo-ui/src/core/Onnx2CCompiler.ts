/* eslint-disable */
// @ts-nocheck
import { WorkerManager } from './WorkerManager';

/**
 * Worker wrapper for ONNX2C compilation logic.
 */
export class Onnx2CCompiler {
  private workerId: string = 'onnx2c';

  /**
   * Compiles an ONNX binary buffer into C source code.
   * @param onnxBuffer The binary contents of the `.onnx` file.
   * @returns A promise resolving to the generated C source code string.
   */
  public async compile(onnxBuffer: Uint8Array): Promise<string> {
    const wm = WorkerManager.getInstance();
    try {
      const cSource = await wm.execute(this.workerId, onnxBuffer, 60000);
      return cSource;
    } catch (e: object) {
      throw new Error(`ONNX2C Compilation failed: ${e.message}`);
    }
  }

  /**
   * Statically calculates memory footprint by parsing the generated C code
   * for standard array declarations and mallocs.
   * @param cSource The generated C source code string.
   * @returns Theoretical memory usage in bytes.
   */
  public static calculateMemoryFootprint(cSource: string): number {
    let bytes = 0;

    // Very naive AST parser mock finding array declarations like: float tensor_abc[100];
    const arrayRegex = /(?:float|int|double)\s+\w+\[(\d+)\]/g;
    let match;
    while ((match = arrayRegex.exec(cSource)) !== null) {
      // Assuming 4 bytes per float/int on average for this mock
      bytes += parseInt(match[1], 10) * 4;
    }

    // Find mallocs: malloc(400)
    const mallocRegex = /malloc\s*\(\s*(\d+)\s*\)/g;
    while ((match = mallocRegex.exec(cSource)) !== null) {
      bytes += parseInt(match[1], 10);
    }

    return bytes;
  }
}
