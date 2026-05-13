import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleSphinxDemoUICommand } from '../src/commands/sphinx-demo-ui.js';
import * as child_process from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

vi.mock('fs', async (importOriginal) => {
  const actual: any = await importOriginal();
  return {
    ...actual,
    existsSync: vi.fn(),
  };
});

vi.mock('child_process', async (importOriginal) => {
  const actual: any = await importOriginal();
  return {
    ...actual,
    spawn: vi.fn(),
  };
});

describe('handleSphinxDemoUICommand', () => {
  let consoleLogSpy: any;
  let consoleErrorSpy: any;
  let processExitSpy: any;
  let cwdSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);
    cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue('/mock/workspace');
  });

  afterEach(() => {
    vi.restoreAllMocks();
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
    vi.mocked(fs.existsSync).mockImplementation((p: any) => {
      if (p.endsWith('pnpm-workspace.yaml')) return true;
      if (p.includes('sphinx-demo-ui')) return true;
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
    vi.mocked(child_process.spawn).mockReturnValue(mockChild as any);

    await handleSphinxDemoUICommand([]);
    expect(consoleLogSpy).toHaveBeenCalledWith('Starting Sphinx Demo UI...');
    expect(child_process.spawn).toHaveBeenCalledWith('pnpm', ['dev'], expect.any(Object));
  });

  it('should exit when ui directory does not exist', async () => {
    vi.mocked(fs.existsSync).mockImplementation((p: any) => {
      if (p.endsWith('pnpm-workspace.yaml')) return true;
      if (p.includes('sphinx-demo-ui')) return false;
      return false;
    });

    try {
      await handleSphinxDemoUICommand([]);
    } catch (e) {}

    expect(consoleErrorSpy).toHaveBeenCalledWith('Sphinx Demo UI not found in monorepo.');
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });
});
