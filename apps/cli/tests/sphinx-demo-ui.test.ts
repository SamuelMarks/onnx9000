import * as child_process from 'node:child_process';
import * as fs from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { handleSphinxDemoUICommand } from '../src/commands/sphinx-demo-ui.js';

vi.mock('node:fs', async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    existsSync: vi.fn(),
  };
});

vi.mock('node:child_process', async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    spawn: vi.fn(),
  };
});

describe('handleSphinxDemoUICommand', () => {
  let consoleLogSpy: ReturnType<typeof vi.spyOn>;
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;
  let processExitSpy: ReturnType<typeof vi.spyOn>;
  let _cwdSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
    _cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue('/mock/workspace');
  });

  afterEach(() => {
    consoleLogSpy.mockRestore();
    consoleErrorSpy.mockRestore();
    processExitSpy.mockRestore();
    _cwdSpy.mockRestore();
    vi.mocked(fs.existsSync).mockReset();
    vi.mocked(child_process.spawn).mockReset();
  });

  it('should print help and exit if -h', async () => {
    await handleSphinxDemoUICommand(['-h']);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining('Usage: onnx9000 sphinx-demo-ui'),
    );
    expect(processExitSpy).toHaveBeenCalledWith(0);
  });

  it('should run pnpm dev when ui directory exists', async () => {
    vi.mocked(fs.existsSync).mockImplementation((p: fs.PathLike) => {
      if (String(p).endsWith('pnpm-workspace.yaml')) return true;
      if (String(p).includes('sphinx-demo-ui')) return true;
      return false;
    });

    const mockChild = {
      on: vi.fn((event, cb) => {
        if (event === 'close') {
          setTimeout(() => cb(0), 10);
        }
      }),
      kill: vi.fn(),
    };
    vi.mocked(child_process.spawn).mockReturnValue(
      mockChild as ReturnType<typeof child_process.spawn>,
    );

    await handleSphinxDemoUICommand([]);
    expect(consoleLogSpy).toHaveBeenCalledWith('Starting Sphinx Demo UI...');
    expect(child_process.spawn).toHaveBeenCalledWith('pnpm', ['dev'], expect.any(Object));
  });

  it('should exit when ui directory does not exist', async () => {
    vi.mocked(fs.existsSync).mockImplementation((p: fs.PathLike) => {
      if (String(p).endsWith('pnpm-workspace.yaml')) return true;
      if (String(p).includes('sphinx-demo-ui')) return false;
      return false;
    });

    try {
      await handleSphinxDemoUICommand([]);
    } catch (_e) {
      expect(_e).toBeDefined();
    }

    expect(consoleErrorSpy).toHaveBeenCalledWith('Sphinx Demo UI directory not found.');
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });

  it('should handle parent directory walk, non-zero exit, error event, and SIGINT', async () => {
    await handleSphinxDemoUICommand(['--help']);
    _cwdSpy.mockReturnValue('/mock/workspace/nested');
    vi.mocked(fs.existsSync).mockImplementation((p: fs.PathLike) => {
      if (String(p) === '/mock/workspace/pnpm-workspace.yaml') return true;
      if (String(p).includes('sphinx-demo-ui')) return true;
      return false;
    });

    let closeCb: ((code: number | null) => void) | undefined;
    let errCb: ((err: Error) => void) | undefined;
    const mockChild = {
      on: vi.fn((event: string, cb: (...args: never[]) => void) => {
        if (event === 'close') closeCb = cb as (code: number | null) => void;
        if (event === 'error') errCb = cb as (err: Error) => void;
      }),
      kill: vi.fn(),
    };
    vi.mocked(child_process.spawn).mockReturnValue(
      mockChild as unknown as ReturnType<typeof child_process.spawn>,
    );

    const p1 = handleSphinxDemoUICommand([]);
    closeCb?.(1);
    await expect(p1).rejects.toThrow('Sphinx Demo UI exited with code 1');

    const p2 = handleSphinxDemoUICommand([]);
    errCb?.(new Error('spawn err'));
    await expect(p2).rejects.toThrow('spawn err');

    const p3 = handleSphinxDemoUICommand([]);
    process.emit('SIGINT');
    await p3;
    expect(mockChild.kill).toHaveBeenCalledWith('SIGINT');
  });
});
