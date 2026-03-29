/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Onnx2CCompiler } from '../../src/core/Onnx2CCompiler';
import { WorkerManager } from '../../src/core/WorkerManager';

describe('Onnx2CCompiler', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('should compile onnx buffer to C source via WorkerManager', async () => {
    const wm = WorkerManager.getInstance();
    const executeSpy = vi.spyOn(wm, 'execute').mockResolvedValue('void main() { return 0; }');

    const compiler = new Onnx2CCompiler();
    const buf = new Uint8Array([1, 2, 3]);

    const res = await compiler.compile(buf);

    expect(executeSpy).toHaveBeenCalledWith('onnx2c', buf, 60000);
    expect(res).toBe('void main() { return 0; }');
  });

  it('should wrap execution errors properly', async () => {
    const wm = WorkerManager.getInstance();
    vi.spyOn(wm, 'execute').mockRejectedValue(new Error('Syntax Error'));

    const compiler = new Onnx2CCompiler();
    const buf = new Uint8Array([1]);

    await expect(compiler.compile(buf)).rejects.toThrow('ONNX2C Compilation failed: Syntax Error');
  });

  it('should accurately calculate memory footprint from static C arrays', () => {
    const cSource = `
      float input[1024];
      int shape[4];
      double bias[128];
    `;
    const mem = Onnx2CCompiler.calculateMemoryFootprint(cSource);
    // (1024 * 4) + (4 * 4) + (128 * 4) = 4096 + 16 + 512 = 4624
    expect(mem).toBe(4624);
  });

  it('should accurately calculate memory footprint from dynamic mallocs', () => {
    const cSource = `
      void* ptr1 = malloc(400);
      void* ptr2 = malloc( 1024 );
    `;
    const mem = Onnx2CCompiler.calculateMemoryFootprint(cSource);
    expect(mem).toBe(1424);
  });

  it('should combine static and dynamic memory calculations', () => {
    const cSource = `
      float input[10]; // 40
      void* ptr1 = malloc(10); // 10
    `;
    const mem = Onnx2CCompiler.calculateMemoryFootprint(cSource);
    expect(mem).toBe(50);
  });

  it('should return 0 for C code with no static arrays or mallocs', () => {
    const cSource = `
      int main() {
        int a = 1;
        float b = 2.0f;
        return a + (int)b;
      }
    `;
    const mem = Onnx2CCompiler.calculateMemoryFootprint(cSource);
    expect(mem).toBe(0);
  });
});
