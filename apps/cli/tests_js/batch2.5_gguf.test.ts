import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleOnnx2GgufCommand, handleGguf2OnnxCommand } from '../src/commands/gguf.js';
import * as core from '@onnx9000/core';
import * as onnx2gguf from '@onnx9000/onnx2gguf';
import * as fs from 'fs';

vi.mock('@onnx9000/core', () => ({
  load: vi.fn().mockResolvedValue({}),
  save: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
}));

vi.mock('@onnx9000/onnx2gguf', () => ({
  compileGGUF: vi.fn().mockReturnValue(new Uint8Array([4, 5, 6])),
  reconstructONNX: vi.fn().mockReturnValue({}),
  GGUFReader: vi.fn().mockImplementation(() => ({})),
}));

vi.mock('fs', () => {
  return {
    ...vi.importActual('fs'),
    statSync: vi.fn(),
    readFileSync: vi.fn(),
    writeFileSync: vi.fn(),
  };
});

describe('CLI Commands Batch 2.5 (GGUF)', () => {
  let consoleLogSpy: any;
  let consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation((code) => {
      throw new Error('Process exited with ' + code);
    });

    vi.mocked(fs.statSync).mockReturnValue({ size: 100 } as any);
    vi.mocked(fs.readFileSync).mockImplementation((path: any) => {
      if (typeof path === 'string' && path.includes('tokenizer')) return '{"mock": true}';
      return Buffer.from([1, 2, 3]) as any;
    });
    vi.mocked(fs.writeFileSync).mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('handleOnnx2GgufCommand', () => {
    it('errors when missing model path', async () => {
      await expect(handleOnnx2GgufCommand([])).rejects.toThrow('Process exited with 1');
      expect(processExitSpy).toHaveBeenCalledWith(1);
    });

    it('handles dry run', async () => {
      await handleOnnx2GgufCommand(['model.onnx', '--dry-run']);
      expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Dry run:'));
    });

    it('warns on massive model without force', async () => {
      vi.mocked(fs.statSync).mockReturnValue({ size: 80_000_000_000 } as any);
      await handleOnnx2GgufCommand(['model.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Warning: Massive model detected.'),
      );
    });

    it('proceeds on massive model with force', async () => {
      vi.mocked(fs.statSync).mockReturnValue({ size: 80_000_000_000 } as any);
      await handleOnnx2GgufCommand(['model.onnx', '--force']);
      expect(fs.writeFileSync).toHaveBeenCalled();
    });

    it('processes command with full args', async () => {
      await handleOnnx2GgufCommand([
        '--tokenizer',
        'tokenizer.json',
        '--outtype',
        'f16',
        '--architecture',
        'llama',
        '-o',
        'out.gguf',
        'model.onnx',
      ]);
      expect(fs.writeFileSync).toHaveBeenCalledWith('out.gguf', expect.any(Uint8Array));
      expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Saved GGUF to out.gguf'));
    });

    it('processes command with alternate args format', async () => {
      await handleOnnx2GgufCommand(['model.onnx', '--output', 'out2.gguf']);
      expect(fs.writeFileSync).toHaveBeenCalledWith('out2.gguf', expect.any(Uint8Array));
    });

    it('processes command with default output', async () => {
      await handleOnnx2GgufCommand(['model.onnx']);
      expect(fs.writeFileSync).toHaveBeenCalledWith('model.gguf', expect.any(Uint8Array));
    });
  });

  describe('handleGguf2OnnxCommand', () => {
    it('errors when missing model path', async () => {
      await expect(handleGguf2OnnxCommand([])).rejects.toThrow('Process exited with 1');
      expect(processExitSpy).toHaveBeenCalledWith(1);
    });

    it('processes command with default output', async () => {
      await handleGguf2OnnxCommand(['model.gguf']);
      expect(fs.writeFileSync).toHaveBeenCalledWith('model.onnx', expect.any(Uint8Array));
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Saved ONNX to model.onnx'),
      );
    });

    it('processes command with custom output -o', async () => {
      await handleGguf2OnnxCommand(['model.gguf', '-o', 'out.onnx']);
      expect(fs.writeFileSync).toHaveBeenCalledWith('out.onnx', expect.any(Uint8Array));
    });

    it('processes command with custom output --output', async () => {
      await handleGguf2OnnxCommand(['model.gguf', '--output', 'out2.onnx']);
      expect(fs.writeFileSync).toHaveBeenCalledWith('out2.onnx', expect.any(Uint8Array));
    });
  });
});
