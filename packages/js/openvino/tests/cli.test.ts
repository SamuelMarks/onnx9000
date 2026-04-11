import { describe, it, expect, vi, beforeEach } from 'vitest';
import fs from 'fs';
import { load } from '@onnx9000/core';
import { main } from '../bin/cli.js';

// Mock core load
vi.mock('@onnx9000/core', async () => {
  const actual = await vi.importActual('@onnx9000/core');
  return {
    ...actual,
    load: vi.fn().mockReturnValue({
      inputs: [{ name: 'in', shape: [1, 3] }],
      nodes: [],
      outputs: [],
    }),
  };
});

// Mock fs
vi.mock('fs', () => ({
  default: {
    readFileSync: vi.fn().mockReturnValue(Buffer.from('mock')),
    writeFileSync: vi.fn(),
    existsSync: vi.fn().mockReturnValue(true),
    mkdirSync: vi.fn(),
  },
}));

// Mock the exporter
vi.mock('../dist/index.js', () => ({
  OpenVinoExporter: vi.fn().mockImplementation(() => ({
    export: vi.fn().mockReturnValue({ xml: '<net></net>', bin: new Uint8Array([0]) }),
  })),
}));

describe('OpenVino CLI', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should show help if no args', () => {
    process.argv = ['node', 'cli.js'];
    const spy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(() => {
      throw new Error('exit');
    });

    expect(() => main()).toThrow('exit');

    expect(spy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    spy.mockRestore();
    exitSpy.mockRestore();
  });

  it('should run export with args and create dir', () => {
    process.argv = ['node', 'cli.js', 'model.onnx', '-o', 'new_out'];
    const spy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
    const existsSpy = vi.spyOn(fs, 'existsSync').mockReturnValue(false);

    main();

    expect(fs.mkdirSync).toHaveBeenCalledWith('new_out', { recursive: true });
    expect(fs.writeFileSync).toHaveBeenCalled();
    spy.mockRestore();
    existsSpy.mockRestore();
  });
});
