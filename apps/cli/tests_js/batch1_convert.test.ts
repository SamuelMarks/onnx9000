import * as fs from 'node:fs';
import { mmdnn } from '@onnx9000/converters';
import { Onnx9000Error } from '@onnx9000/core';
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
  let consoleLogSpy: ReturnType<typeof vi.spyOn>;
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;
  let processExitSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation((code) => {
      throw new Onnx9000Error(`Process exited with ${code}`);
    });

    vi.mocked(fs.statSync).mockImplementation(((path: fs.PathLike) => {
      if (String(path) === 'dir') {
        return { isDirectory: () => true, isFile: () => false, size: 0 } as fs.Stats;
      }
      if (!path) return undefined;
      return { isDirectory: () => false, isFile: () => true, size: 100 } as fs.Stats;
    }) as typeof fs.statSync);
    vi.mocked(fs.readdirSync).mockReturnValue(['file1.h5', 'file2.h5'] as ReturnType<
      typeof fs.readdirSync
    >);
    vi.mocked(fs.writeFileSync).mockImplementation(() => {});
    vi.mocked(fs.promises.readFile).mockResolvedValue(Buffer.from([1, 2, 3]));
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
    const lastCall = mockCalls[mockCalls.length - 1];
    if (lastCall) {
      const blobs = lastCall[2] as Blob[];
      const firstBlob = blobs[0];
      if (firstBlob) {
        firstBlob.stream();
        await firstBlob.arrayBuffer();
        firstBlob.slice();
        firstBlob.slice(1, 10);
      }
    }
  });

  it('handles result as object', async () => {
    vi.mocked(mmdnn.convert).mockResolvedValueOnce({} as Awaited<ReturnType<typeof mmdnn.convert>>);

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
