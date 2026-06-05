import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleCudaCommand } from '../src/commands/cuda.js';
import { handleDiffusersCommand } from '../src/commands/diffusers.js';
import { handleEditCommand } from '../src/commands/edit.js';
import { handleExportCommand } from '../src/commands/export.js';
import { handleHummingbirdCommand } from '../src/commands/hummingbird.js';
import { handleInspectCommand } from '../src/commands/inspect.js';
import * as child_process from 'child_process';
import * as fs from 'fs';

vi.mock('child_process', () => ({
  spawn: vi.fn(),
}));

vi.mock('fs', () => {
  return {
    ...vi.importActual('fs'),
    existsSync: vi.fn(),
    writeFileSync: vi.fn(),
  };
});

describe('CLI Commands Batch 2', () => {
  let consoleLogSpy: any;
  let consoleErrorSpy: any;
  let processExitSpy: any;
  let processOnSpy: any;
  let processCwdSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
    processOnSpy = vi.spyOn(process, 'on').mockImplementation(() => process as any);
    processCwdSpy = vi.spyOn(process, 'cwd').mockReturnValue('/fake/path');

    vi.mocked(fs.existsSync).mockReturnValue(true);
    vi.mocked(fs.writeFileSync).mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('handleCudaCommand', () => {
    it('shows help when no args or -h', () => {
      handleCudaCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleCudaCommand(['-h']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('processes command', () => {
      handleCudaCommand(['model.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Initializing CUDA execution for model.onnx'),
      );
      expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('CUDA engine loaded.'));
      // fallback
      handleCudaCommand(['']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Initializing CUDA execution for '),
      );
    });
  });

  describe('handleDiffusersCommand', () => {
    it('shows help when invalid args', () => {
      handleDiffusersCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleDiffusersCommand(['model', '--invalid']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleDiffusersCommand(['-h', '--prompt', 'cat']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleDiffusersCommand(['--help', '--prompt', 'cat']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleDiffusersCommand(['model', 'not--prompt', 'cat']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('processes generation without output file', () => {
      handleDiffusersCommand(['model.onnx', '--prompt', 'cat']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Generated image tensor successfully'),
      );
      // cover empty args boundaries
      handleDiffusersCommand(['', '--prompt', 'cat']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Initializing Diffusion Pipeline from: ...'),
      );
    });

    it('processes generation with output file via -o', () => {
      vi.mocked(fs.writeFileSync).mockClear();
      handleDiffusersCommand(['model.onnx', '--prompt', 'cat', '-o', 'out.png']);
      expect(fs.writeFileSync).toHaveBeenCalledWith('out.png', expect.any(String));
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Image tensor saved to out.png'),
      );
    });

    it('processes generation with output file via --output', () => {
      vi.mocked(fs.writeFileSync).mockClear();
      handleDiffusersCommand(['model.onnx', '--prompt', 'cat', '--output', 'out.png']);
      expect(fs.writeFileSync).toHaveBeenCalledWith('out.png', expect.any(String));
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Image tensor saved to out.png'),
      );
      // boundary
      handleDiffusersCommand(['model.onnx', '--prompt', 'cat', '--output']);
      handleDiffusersCommand(['model.onnx', '--prompt', 'cat', '-o']);
      // third argument branch for outputpath missing
      handleDiffusersCommand(['model.onnx', '--prompt', '']);
    });
  });

  describe('handleEditCommand', () => {
    it('fails when ui dir not found', async () => {
      vi.mocked(fs.existsSync).mockReturnValue(false);
      await handleEditCommand(['model.onnx']);
      expect(processExitSpy).toHaveBeenCalledWith(1);
      // test empty arg
      await handleEditCommand([]);
    });

    it('spawns pnpm dev when ui dir exists', async () => {
      vi.mocked(fs.existsSync).mockImplementation((p: any) => {
        if (p.includes('pnpm-workspace.yaml')) return true;
        if (p.includes('netron-ui')) return true;
        return false;
      });

      let onCb: any;
      const mockChild = {
        on: vi.fn().mockImplementation((event, cb) => {
          if (event === 'close') onCb = cb;
          return mockChild;
        }),
        kill: vi.fn(),
      };
      vi.mocked(child_process.spawn).mockReturnValue(mockChild as any);

      const promise = handleEditCommand(['model.onnx']);
      expect(child_process.spawn).toHaveBeenCalledWith('pnpm', ['dev'], expect.any(Object));

      onCb(0);
      await promise;
    });

    it('rejects when process exits with error', async () => {
      vi.mocked(fs.existsSync).mockReturnValue(true);

      let onCb: any;
      const mockChild = {
        on: vi.fn().mockImplementation((event, cb) => {
          if (event === 'close') onCb = cb;
          return mockChild;
        }),
        kill: vi.fn(),
      };
      vi.mocked(child_process.spawn).mockReturnValue(mockChild as any);

      const promise = handleEditCommand(['model.onnx']);
      onCb(1);

      await expect(promise).rejects.toThrow('Modifier UI exited with code 1');
    });

    it('rejects on child error', async () => {
      vi.mocked(fs.existsSync).mockReturnValue(true);

      let errCb: any;
      const mockChild = {
        on: vi.fn().mockImplementation((event, cb) => {
          if (event === 'error') errCb = cb;
          return mockChild;
        }),
        kill: vi.fn(),
      };
      vi.mocked(child_process.spawn).mockReturnValue(mockChild as any);

      const promise = handleEditCommand(['model.onnx']);
      errCb(new Error('child err'));

      await expect(promise).rejects.toThrow('child err');
    });

    it('handles SIGINT', async () => {
      vi.mocked(fs.existsSync).mockReturnValue(true);

      const mockChild = {
        on: vi.fn().mockReturnValue({}),
        kill: vi.fn(),
      };
      vi.mocked(child_process.spawn).mockReturnValue(mockChild as any);

      let sigintCb: any;
      processOnSpy.mockImplementation((event: string, cb: any) => {
        if (event === 'SIGINT') sigintCb = cb;
        return process;
      });

      const promise = handleEditCommand(['model.onnx']);
      sigintCb();
      await promise;
      expect(mockChild.kill).toHaveBeenCalledWith('SIGINT');
    });
  });

  describe('handleExportCommand', () => {
    it('shows help when no args or -h', () => {
      handleExportCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleExportCommand(['-h']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('errors on unsupported format', () => {
      handleExportCommand(['model.onnx', '--format', 'invalid']);
      expect(processExitSpy).toHaveBeenCalledWith(1);
      handleExportCommand(['model.onnx', '--format']);
      expect(processExitSpy).toHaveBeenCalledWith(1);
    });

    it('processes c export with default output', () => {
      handleExportCommand(['model.onnx', '--format', 'c']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Saving C source to model.c'),
      );
      handleExportCommand(['', '--format', 'c']);
    });

    it('processes c export with custom output', () => {
      handleExportCommand(['model.onnx', '--format', 'c', '-o', 'out.c']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Saving C source to out.c'),
      );
      handleExportCommand(['model.onnx', '--format', 'c', '-o']);
    });
  });

  describe('handleHummingbirdCommand', () => {
    it('shows help when no args or -h', () => {
      handleHummingbirdCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleHummingbirdCommand(['-h']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('processes with default output', () => {
      handleHummingbirdCommand(['model.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Saving optimized tensor model to model_tensor.onnx'),
      );
      handleHummingbirdCommand(['']);
    });

    it('processes with custom output', () => {
      handleHummingbirdCommand(['model.onnx', '-o', 'out.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Saving optimized tensor model to out.onnx'),
      );
      handleHummingbirdCommand(['model.onnx', '-o']);
    });
  });

  describe('handleInspectCommand', () => {
    it('errors when no args', async () => {
      try {
        await handleInspectCommand([]);
      } catch (e) {}
      expect(processExitSpy).toHaveBeenCalledWith(1);
    });

    it('errors when file not found', async () => {
      vi.mocked(fs.existsSync).mockReturnValue(false);
      try {
        await handleInspectCommand(['model.keras']);
      } catch (e) {}
      expect(processExitSpy).toHaveBeenCalledWith(1);
    });

    it('inspects valid keras model', async () => {
      vi.mocked(fs.existsSync).mockReturnValue(true);
      await handleInspectCommand(['model.keras']);
      expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Model Summary:'));
    });

    it('handles unsupported format', async () => {
      vi.mocked(fs.existsSync).mockReturnValue(true);
      await handleInspectCommand(['model.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Unsupported format'));
    });
  });
});
