import * as fs from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { handleWebnnPolyfillCommand } from '../src/commands/webnn-polyfill.js';
import { handleWhisperLlmCommand } from '../src/commands/whisper-llm.js';
import { handleZeroDepClassifierCommand } from '../src/commands/zero-dep-classifier.js';
import { handleZooCommand } from '../src/commands/zoo.js';

vi.mock('fs', () => ({
  default: {
    existsSync: vi.fn(),
    writeFileSync: vi.fn(),
  },
  existsSync: vi.fn(),
  writeFileSync: vi.fn(),
}));

vi.mock('@onnx9000/core', () => ({
  Whisper: class {},
}));

vi.mock('@onnx9000/agent', () => ({
  validateZooModel: vi.fn().mockResolvedValue(undefined),
}));

describe('CLI Commands Batch 6', () => {
  let consoleLogSpy: any;
  let _consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    _consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
    vi.mocked(fs.writeFileSync).mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('handleWebnnPolyfillCommand', () => {
    it('shows help when -h', () => {
      handleWebnnPolyfillCommand(['-h']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('processes command', () => {
      handleWebnnPolyfillCommand([]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Testing WebNN Polyfill compatibility...'),
      );
    });
  });

  describe('handleWhisperLlmCommand', () => {
    it('shows help when missing args or -h', () => {
      handleWhisperLlmCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleWhisperLlmCommand(['-h']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('processes command without output file', () => {
      handleWhisperLlmCommand(['model.onnx', 'audio.wav']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Transcribing audio.wav...'),
      );
      handleWhisperLlmCommand(['', '']);
    });

    it('processes command with custom output -o', () => {
      handleWhisperLlmCommand(['model.onnx', 'audio.wav', '-o', 'out.txt']);
      expect(fs.writeFileSync).toHaveBeenCalledWith('out.txt', 'Transcribed text mock');
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Transcription saved to out.txt'),
      );
      handleWhisperLlmCommand(['model.onnx', 'audio.wav', '-o']);
      handleWhisperLlmCommand(['model.onnx', 'audio.wav', '--output']);
    });
  });

  describe('handleZeroDepClassifierCommand', () => {
    it('shows help when no args or -h', () => {
      handleZeroDepClassifierCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleZeroDepClassifierCommand(['-h']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('processes command', () => {
      handleZeroDepClassifierCommand(['model.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Generating zero-dependency classifier for model.onnx...'),
      );
      handleZeroDepClassifierCommand(['']);
    });
  });

  describe('handleZooCommand', () => {
    it('shows help when no args or -h', () => {
      handleZooCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleZooCommand(['-h']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    describe('pull command', () => {
      it('errors when missing model_id', () => {
        handleZooCommand(['pull']);
        expect(processExitSpy).toHaveBeenCalledWith(1);
      });

      it('processes pull', () => {
        handleZooCommand(['pull', 'model123']);
        expect(consoleLogSpy).toHaveBeenCalledWith(
          expect.stringContaining('Downloading model123...'),
        );
      });

      it('handles string fallback logic', () => {
        const mockModelId = { toString: () => '' } as any;
        Object.defineProperty(mockModelId, 'length', { value: 1 });
        handleZooCommand(['pull', mockModelId]);
      });
    });

    it('errors on unknown command', () => {
      handleZooCommand(['unknown']);
      expect(processExitSpy).toHaveBeenCalledWith(1);
      handleZooCommand(['']);
    });
  });
});
