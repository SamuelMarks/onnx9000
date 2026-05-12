import { describe, it, expect, vi } from 'vitest';
import { handleJsonExtractCommand } from '../src/commands/json-extract.js';
import * as fs from 'fs';
import * as core from '@onnx9000/core';

vi.mock('fs');
vi.mock('@onnx9000/core');

describe('handleJsonExtractCommand', () => {
  it('should print help and exit if no args or -h', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);

    await handleJsonExtractCommand([]);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    await handleJsonExtractCommand(['-h']);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    consoleSpy.mockRestore();
    exitSpy.mockRestore();
  });

  it('should extract and write json to stdout', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(fs, 'readFileSync').mockReturnValue(Buffer.from('dummy') as any);
    vi.mocked(core.load).mockResolvedValue({ myAttr: 'value' } as any);

    await handleJsonExtractCommand(['dummy.onnx']);

    expect(core.load).toHaveBeenCalled();
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('"myAttr": "value"'));

    consoleSpy.mockRestore();
  });

  it('should extract and write json to file', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(fs, 'readFileSync').mockReturnValue(Buffer.from('dummy') as any);
    const writeFileSyncSpy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {});
    vi.mocked(core.load).mockResolvedValue({
      myAttr: new Uint8Array([1, 2, 3]),
      data: new Uint8Array([4]),
      num: 42n,
    } as any);

    await handleJsonExtractCommand(['dummy.onnx', '-o', 'out.json']);

    expect(core.load).toHaveBeenCalled();
    expect(writeFileSyncSpy).toHaveBeenCalled();

    // The second arg to writeFileSync is the JSON string
    const jsonOutput = writeFileSyncSpy.mock.calls[0][1] as string;
    expect(jsonOutput).toContain('[Buffer: 1 bytes]');
    expect(jsonOutput).toContain('42n');

    consoleSpy.mockRestore();
  });
});
