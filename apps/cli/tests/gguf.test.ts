import * as fs from 'node:fs';
import * as core from '@onnx9000/core';
import * as onnx2gguf from '@onnx9000/onnx2gguf';
import { expect, test, vi } from 'vitest';
import { handleGguf2OnnxCommand, handleOnnx2GgufCommand } from '../src/commands/gguf.js';

vi.mock('fs');
vi.mock('@onnx9000/core', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@onnx9000/core')>();
  return { ...actual, load: vi.fn(), save: vi.fn() };
});
vi.mock('@onnx9000/onnx2gguf');

test('onnx2gguf dry run', async () => {
  const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
  await handleOnnx2GgufCommand(['dummy.onnx', '--dry-run']);
  expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('Dry run'));
  logSpy.mockRestore();
});

test('onnx2gguf massive', async () => {
  const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
  vi.spyOn(fs, 'statSync').mockReturnValue({ size: 80_000_000_000 } as fs.Stats);
  await handleOnnx2GgufCommand(['dummy.onnx']);
  expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('Warning: Massive model detected.'));
  logSpy.mockRestore();
});

test('onnx2gguf full execution', async () => {
  vi.spyOn(fs, 'statSync').mockReturnValue({ size: 100 } as fs.Stats);
  vi.spyOn(fs, 'readFileSync').mockReturnValue(Buffer.from('mock'));
  const writeSpy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {});
  vi.mocked(core.load).mockResolvedValue(new core.Graph('graph'));
  vi.spyOn(onnx2gguf, 'compileGGUF').mockReturnValue(new ArrayBuffer(10));

  await handleOnnx2GgufCommand([
    'dummy.onnx',
    '-o',
    'dummy.gguf',
    '--tokenizer',
    'tok.json',
    '--outtype',
    'f16',
    '--architecture',
    'llama',
    '--force',
  ]);
  expect(writeSpy).toHaveBeenCalled();
});

test('gguf2onnx execution', async () => {
  const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {
    throw new Error('exit');
  }) as never);
  await expect(handleOnnx2GgufCommand([])).rejects.toThrow('exit');
  await expect(handleGguf2OnnxCommand([])).rejects.toThrow('exit');
  exitSpy.mockRestore();
  errSpy.mockRestore();

  vi.spyOn(fs, 'statSync').mockReturnValue({ size: 100 } as fs.Stats);
  vi.spyOn(fs, 'readFileSync').mockReturnValue(Buffer.from('mock'));
  const writeSpy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {});
  vi.mocked(core.load).mockResolvedValue(new core.Graph('graph'));
  vi.spyOn(onnx2gguf, 'compileGGUF').mockReturnValue(new ArrayBuffer(10));
  await handleOnnx2GgufCommand(['dummy.onnx']);

  vi.spyOn(onnx2gguf, 'GGUFReader').mockImplementation(
    () => ({}) as InstanceType<typeof onnx2gguf.GGUFReader>,
  );
  vi.spyOn(onnx2gguf, 'reconstructONNX').mockReturnValue(new core.Graph('graph'));
  vi.mocked(core.save).mockResolvedValue(new Uint8Array(10));

  await handleGguf2OnnxCommand(['dummy.gguf', '-o', 'dummy.onnx']);
  await handleGguf2OnnxCommand(['dummy.gguf', '--output', 'dummy2.onnx', '--flag']);
  await handleGguf2OnnxCommand(['dummy.gguf']);
  expect(writeSpy).toHaveBeenCalled();
});
