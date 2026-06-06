import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { handleWasmCommand } from '../src/commands/wasm.js';
import { handleWebgpuCommand } from '../src/commands/webgpu.js';

describe('CLI Commands Batch 5', () => {
  let consoleLogSpy: any;
  let _consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    _consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('handleWasmCommand', () => {
    it('shows help when no args or -h', () => {
      handleWasmCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleWasmCommand(['-h']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('processes command', () => {
      handleWasmCommand(['model.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Initializing WebAssembly execution for model.onnx'),
      );
      handleWasmCommand(['']);
    });
  });

  describe('handleWebgpuCommand', () => {
    it('shows help when no args or -h', () => {
      handleWebgpuCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleWebgpuCommand(['-h']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('processes command', () => {
      handleWebgpuCommand(['model.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Initializing WebGPU execution for model.onnx'),
      );
      handleWebgpuCommand(['']);
    });
  });
});
