import * as fs from 'node:fs';
import * as core from '@onnx9000/core';
import { describe, expect, it, vi } from 'vitest';
import { handlePytorchCodegenCommand } from '../src/commands/pytorch-codegen.js';

vi.mock('fs');
vi.mock('@onnx9000/core', async (importOriginal) => {
  const mod = await importOriginal<typeof import('@onnx9000/core')>();
  return {
    ...mod,
    load: vi.fn(),
    ONNXToPyTorchVisitor: vi.fn(),
  };
});

describe('handlePytorchCodegenCommand', () => {
  it('should print help and exit if no args or -h', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);

    await handlePytorchCodegenCommand([]);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    await handlePytorchCodegenCommand(['-h']);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    consoleSpy.mockRestore();
    exitSpy.mockRestore();
  });

  it('should extract and write code to stdout', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(fs, 'readFileSync').mockReturnValue(Buffer.from('dummy'));
    const mockGraph = new core.Graph('test');
    vi.mocked(core.load).mockResolvedValue(mockGraph);

    const mockGenerate = vi.fn().mockReturnValue('class MockCode: pass');
    vi.mocked(core.ONNXToPyTorchVisitor).mockImplementation(() => {
      return { generate: mockGenerate } as InstanceType<typeof core.ONNXToPyTorchVisitor>;
    });

    await handlePytorchCodegenCommand(['dummy.onnx']);

    expect(core.load).toHaveBeenCalled();
    expect(core.ONNXToPyTorchVisitor).toHaveBeenCalledWith(mockGraph);
    expect(consoleSpy).toHaveBeenCalledWith('class MockCode: pass');

    consoleSpy.mockRestore();
  });

  it('should extract and write code to file', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(fs, 'readFileSync').mockReturnValue(Buffer.from('dummy'));
    const writeFileSyncSpy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {});
    const mockGraph = new core.Graph('test');
    vi.mocked(core.load).mockResolvedValue(mockGraph);

    const mockGenerate = vi.fn().mockReturnValue('class MockCode: pass');
    vi.mocked(core.ONNXToPyTorchVisitor).mockImplementation(() => {
      return { generate: mockGenerate } as InstanceType<typeof core.ONNXToPyTorchVisitor>;
    });

    await handlePytorchCodegenCommand(['dummy.onnx', '-o', 'out.py']);

    expect(core.load).toHaveBeenCalled();
    expect(writeFileSyncSpy).toHaveBeenCalledWith('out.py', 'class MockCode: pass');

    consoleSpy.mockRestore();
  });
});
