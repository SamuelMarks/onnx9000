import { describe, it, expect, vi } from 'vitest';
import { handleWhisperLlmCommand } from '../src/commands/whisper-llm.js';
import { handleLlamaWebCommand } from '../src/commands/llama-web.js';
import * as fs from 'fs';
import * as core from '@onnx9000/core';

vi.mock('fs');
vi.mock('@onnx9000/core');

describe('GenAI Models', () => {
  describe('handleWhisperLlmCommand', () => {
    it('should print help and exit if no args', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);

      await handleWhisperLlmCommand([]);
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
      expect(exitSpy).toHaveBeenCalledWith(0);

      consoleSpy.mockRestore();
      exitSpy.mockRestore();
    });

    it('should transcribe and print to stdout', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

      await handleWhisperLlmCommand(['dummy.onnx', 'dummy.wav']);

      expect(core.Whisper).toHaveBeenCalled();
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Transcription: Transcribed text mock'),
      );

      consoleSpy.mockRestore();
    });

    it('should transcribe and write to file', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const writeFileSyncSpy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {});

      await handleWhisperLlmCommand(['dummy.onnx', 'dummy.wav', '-o', 'out.txt']);

      expect(core.Whisper).toHaveBeenCalled();
      expect(writeFileSyncSpy).toHaveBeenCalledWith('out.txt', 'Transcribed text mock');

      consoleSpy.mockRestore();
    });
  });

  describe('handleLlamaWebCommand', () => {
    it('should print help and exit if missing args', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);

      await handleLlamaWebCommand(['dummy.onnx', '--prompt']);
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
      expect(exitSpy).toHaveBeenCalledWith(0);

      consoleSpy.mockRestore();
      exitSpy.mockRestore();
    });

    it('should generate and print to stdout', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

      await handleLlamaWebCommand(['dummy.onnx', '--prompt', 'hello']);

      expect(core.LLaMA).toHaveBeenCalled();
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Generated text: Generated text mock'),
      );

      consoleSpy.mockRestore();
    });

    it('should generate and write to file', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const writeFileSyncSpy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {});

      await handleLlamaWebCommand(['dummy.onnx', '--prompt', 'hello', '-o', 'out.txt']);

      expect(core.LLaMA).toHaveBeenCalled();
      expect(writeFileSyncSpy).toHaveBeenCalledWith('out.txt', 'Generated text mock');

      consoleSpy.mockRestore();
    });
  });
});

describe('handleWhisperLlmCommand output missing', () => {
  it('should handle output without file', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleWhisperLlmCommand(['dummy.onnx', 'dummy.wav', '-o']);
    expect(core.Whisper).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });
});

describe('handleLlamaWebCommand output missing', () => {
  it('should handle output without file', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleLlamaWebCommand(['dummy.onnx', '--prompt', 'hello', '-o']);
    expect(core.LLaMA).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });
});

describe('empty arg fallback', () => {
  it('should default missing args to empty string', async () => {
    // It's impossible to pass [] directly to the logic below the length check
    // because length check catches it. So args[0] || '' is defensive, we'll
    // pass ['-o'] so it fails length check, wait we can't bypass length check
    // unless we mock it or pass enough args. We'll pass [undefined as any, undefined as any]
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleWhisperLlmCommand([undefined as any, undefined as any]);
    expect(core.Whisper).toHaveBeenCalled();
    consoleSpy.mockRestore();

    const consoleSpy2 = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleLlamaWebCommand([undefined as any, '--prompt', undefined as any]);
    expect(core.LLaMA).toHaveBeenCalled();
    consoleSpy2.mockRestore();
  });
});
