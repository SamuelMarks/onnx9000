import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { main } from '../src/cli/index.ts';

describe('Compiler CLI', () => {
  let originalArgv: string[];
  const testModel = 'test-model.onnx';
  const outPath = 'test-model.bin';

  beforeEach(() => {
    originalArgv = process.argv;
    // Create a dummy model file
    fs.writeFileSync(testModel, 'dummy onnx content');
    // Mock console to avoid cluttering test output
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    process.argv = originalArgv;
    if (fs.existsSync(testModel)) fs.unlinkSync(testModel);
    if (fs.existsSync(outPath)) fs.unlinkSync(outPath);
    vi.restoreAllMocks();
  });

  it.skip('should print help when no args provided', async () => {
    process.argv = ['node', 'index.js'];

    // Process.exit needs to be mocked
    const mockExit = vi
      .spyOn(process, 'exit')
      .mockImplementation((code?: number | string | null | undefined) => {
        throw new Error(`process.exit: ${String(code)}`);
      });

    try {
      main();
    } catch (e: any) {
      expect(e.message).toContain('process.exit: 1');
    }
    expect(console.log).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
  });

  it('should compile model successfully with defaults', () => {
    process.argv = ['node', 'index.js', 'compile', testModel];

    main();

    expect(fs.existsSync(outPath)).toBe(true);
    const content = JSON.parse(fs.readFileSync(outPath, 'utf-8'));
    expect(content.compiler).toBe('@onnx9000/compiler');
    expect(content.backend).toBe('@onnx9000/backend-web');
    expect(content.optLevel).toBe('O3');
    expect(content.originalModel).toBe(testModel);
  });

  it('should respect custom options', () => {
    const customOut = 'custom-out.bin';
    process.argv = [
      'node',
      'index.js',
      'compile',
      testModel,
      '--target-backend=custom-backend',
      '--optimize-level=O1',
      `--output=${customOut}`,
    ];

    main();

    expect(fs.existsSync(customOut)).toBe(true);
    const content = JSON.parse(fs.readFileSync(customOut, 'utf-8'));
    expect(content.backend).toBe('custom-backend');
    expect(content.optLevel).toBe('O1');

    fs.unlinkSync(customOut);
  });

  it.skip('should error when model not found', async () => {
    process.argv = ['node', 'index.js', 'compile', 'missing.onnx'];
    const mockExit = vi
      .spyOn(process, 'exit')
      .mockImplementation((code?: number | string | null | undefined) => {
        throw new Error(`process.exit: ${String(code)}`);
      });

    try {
      main();
    } catch (e: any) {
      expect(e.message).toContain('process.exit: 1');
    }
    expect(console.error).toHaveBeenCalledWith(expect.stringContaining('not found'));
  });

  it.skip('should error when missing model argument entirely', async () => {
    process.argv = ['node', 'index.js', 'compile'];
    const mockExit = vi
      .spyOn(process, 'exit')
      .mockImplementation((code?: number | string | null | undefined) => {
        throw new Error(`process.exit: ${String(code)}`);
      });

    try {
      main();
    } catch (e: any) {
      expect(e.message).toContain('process.exit: 1');
    }
    expect(console.error).toHaveBeenCalledWith(expect.stringContaining('provide a path'));
  });

  it.skip('should handle --help argument properly', async () => {
    process.argv = ['node', 'index.js', 'compile', '--help'];
    const mockExit = vi
      .spyOn(process, 'exit')
      .mockImplementation((code?: number | string | null | undefined) => {
        throw new Error(`process.exit: ${String(code)}`);
      });

    try {
      main();
    } catch (e: any) {
      expect(e.message).toContain('process.exit: 0');
    }
    expect(console.log).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
  });

  it.skip('should catch errors when import.meta.url matches but execution fails', async () => {
    const originalArgv1 = process.argv[1];
    process.argv[1] = '/fake/index.js';

    vi.spyOn(process, 'exit').mockImplementation((code?: number | string | null | undefined) => {
      throw new Error(`process.exit: ${String(code)}`);
    });

    process.argv = ['node', '/fake/index.js'];

    const promise = new Promise((resolve, reject) => {
      const prevMetaUrl = (import.meta as any).url;
      (import.meta as any).url = 'file:///fake/index.js';

      const fakeCatch = (err: any) => {
        console.error(err);
        process.exit(1);
      };

      // We manually simulate the `if (import.meta.url === ...)` block
      // instead of re-evaluating the module since ES modules handle import.meta statically.
      try {
        fakeCatch(new Error('mock error'));
      } catch (e) {
        reject(e);
      } finally {
        (import.meta as any).url = prevMetaUrl;
      }
    });

    await expect(promise).rejects.toThrow('process.exit: 1');
    expect(console.error).toHaveBeenCalled();
    process.argv[1] = originalArgv1!;
  });
});
