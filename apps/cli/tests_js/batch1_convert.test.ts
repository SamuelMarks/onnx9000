import * as fs from 'node:fs';
import { mmdnn } from '@onnx9000/converters';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('fs', () => {
  return {
    ...vi.importActual('fs'),
    statSync: vi.fn(),
    readdirSync: vi.fn(),
    createReadStream: vi.fn(),
    writeFileSync: vi.fn(),
    promises: {
      readFile: vi.fn(),
    },
  };
});

vi.mock('@onnx9000/converters', () => ({
  mmdnn: {
    convert: vi.fn().mockResolvedValue('converted_data'),
  },
}));

import { handleConvertCommand } from '../src/commands/convert.js';

describe('handleConvertCommand coverage', () => {
  let consoleLogSpy: any;
  let consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation((code) => {
      throw new Error(`Process exited with ${code}`);
    });

    vi.mocked(fs.statSync).mockImplementation((path: any) => {
      if (path === 'dir') return { isDirectory: () => true, isFile: () => false, size: 0 } as any;
      if (!path) return undefined as any;
      return { isDirectory: () => false, isFile: () => true, size: 100 } as any;
    });
    vi.mocked(fs.readdirSync).mockReturnValue(['file1.h5', 'file2.h5'] as any);
    vi.mocked(fs.writeFileSync).mockImplementation(() => {});
    vi.mocked(fs.promises.readFile).mockResolvedValue(new Uint8Array([1, 2, 3]) as any);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('handles missing args', async () => {
    await expect(handleConvertCommand([])).rejects.toThrow('Process exited with 1');
    expect(processExitSpy).toHaveBeenCalledWith(1);
    processExitSpy.mockClear();

    await expect(handleConvertCommand(['--src', 'keras'])).rejects.toThrow('Process exited with 1');
    expect(processExitSpy).toHaveBeenCalledWith(1);
    processExitSpy.mockClear();

    await expect(handleConvertCommand(['--src', 'keras', '--dst', 'onnx'])).rejects.toThrow(
      'Process exited with 1',
    );
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });

  it('handles batch and single file conversions', async () => {
    await handleConvertCommand(['--src', 'keras', '--dst', 'onnx', 'dir']);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining('Starting batch conversion'),
    );

    vi.mocked(mmdnn.convert).mockResolvedValueOnce('string_result_output');
    await handleConvertCommand(['--from', 'keras', '--to', 'onnx', 'file.h5']);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Converting file.h5'));
    expect(fs.writeFileSync).toHaveBeenCalledWith('file_converted.out', 'string_result_output');

    // hit the stream arrayBuffer and slice parts inside convert blobs map
    const mockConvert = vi.mocked(mmdnn.convert);
    const mockCalls = mockConvert.mock.calls;
    if (mockCalls.length > 0) {
      const blobs = mockCalls[mockCalls.length - 1][2] as any;
      if (blobs && blobs.length > 0) {
        blobs[0].stream();
        await blobs[0].arrayBuffer();
        blobs[0].slice();
        blobs[0].slice(1, 10);
      }
    }
  });

  it('handles result as object', async () => {
    vi.mocked(mmdnn.convert).mockResolvedValueOnce({} as any);

    await handleConvertCommand(['--from', 'keras', '--to', 'onnx', 'file.h5']);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining('Result is an object/graph. Skipping write for now.'),
    );
  });

  it('handles failing conversion', async () => {
    vi.mocked(mmdnn.convert).mockRejectedValueOnce(new Error('Conversion error'));

    await handleConvertCommand(['--from', 'keras', '--to', 'onnx', 'file.h5']);
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.stringContaining('Conversion failed for file.h5:'),
      expect.any(Error),
    );
  });
});
