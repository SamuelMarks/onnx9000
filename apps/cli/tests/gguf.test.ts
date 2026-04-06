import { expect, test, vi } from 'vitest';
import { handleOnnx2GgufCommand, handleGguf2OnnxCommand } from '../src/commands/gguf';
import * as fs from 'fs';
import * as onnx2gguf from '@onnx9000/onnx2gguf';
import { load, save } from '@onnx9000/core';
import * as core from '@onnx9000/core';

vi.mock('fs');
vi.mock('@onnx9000/core', async () => ({ load: vi.fn(), save: vi.fn() }));
vi.mock('@onnx9000/onnx2gguf');

test('onnx2gguf dry run', async () => {
  const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  await handleOnnx2GgufCommand(['dummy.onnx', '--dry-run']);
  expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('Dry run'));
  logSpy.mockRestore();
});

test('onnx2gguf massive', async () => {
  vi.spyOn(fs, 'statSync').mockReturnValue({ size: 80_000_000_000 } as Object);
  const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  await handleOnnx2GgufCommand(['dummy.onnx']);
  expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('Massive model detected'));
  logSpy.mockRestore();
});

test('onnx2gguf compile', async () => {
  vi.spyOn(fs, 'statSync').mockReturnValue({ size: 100 } as Object);
  vi.spyOn(fs, 'readFileSync').mockReturnValue(Buffer.from(''));
  const writeSpy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {});
  vi.mocked(core.load).mockResolvedValue('graph' as Object);
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

test('gguf2onnx reconstruct', async () => {
  vi.spyOn(fs, 'readFileSync').mockReturnValue(Buffer.from(''));
  const writeSpy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {});
  vi.spyOn(onnx2gguf, 'GGUFReader').mockImplementation(() => ({}) as Object);
  vi.spyOn(onnx2gguf, 'reconstructONNX').mockReturnValue('graph' as Object);
  vi.mocked(core.save).mockResolvedValue(new ArrayBuffer(10) as Object);

  await handleGguf2OnnxCommand(['dummy.gguf', '-o', 'dummy.onnx']);
  expect(writeSpy).toHaveBeenCalled();
});
