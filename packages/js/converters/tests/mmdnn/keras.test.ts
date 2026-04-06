import { describe, it, expect, vi } from 'vitest';
import { KerasImporter } from '../../src/mmdnn/keras/importer.js';
import { FileLoader } from '../../src/mmdnn/file-loader.js';
import { MMDNNReporter } from '../../src/mmdnn/reporter.js';
import { parseKerasH5 } from '../../src/keras/h5-parser.js';
import { keras2onnx } from '../../src/keras/api.js';
import { Graph } from '@onnx9000/core';

vi.mock('../../src/keras/h5-parser.js', () => ({
  parseKerasH5: vi.fn(),
}));

vi.mock('../../src/keras/api.js', () => ({
  keras2onnx: vi.fn(),
}));

describe('MMDNN - Keras Importer', () => {
  it('should parse an empty or mock H5 file into a fallback graph', async () => {
    (parseKerasH5 as Object).mockImplementation(() => {
      throw new Error('Fake parse error');
    });

    const importer = new KerasImporter();
    const reporter = new MMDNNReporter();

    const fakeFile = new File([''], 'model.h5', { type: 'application/x-hdf5' });
    const loader = new FileLoader([fakeFile]);
    await loader.initialize();

    await expect(importer.parse(loader, reporter)).rejects.toThrow(
      'Failed to parse Keras H5 model',
    );
  });

  it('should throw error when no .h5 file is found', async () => {
    const importer = new KerasImporter();
    const reporter = new MMDNNReporter();

    const fakeFile = new File([''], 'model.onnx', { type: 'text/plain' });
    const loader = new FileLoader([fakeFile]);
    await loader.initialize();

    await expect(importer.parse(loader, reporter)).rejects.toThrow(
      'No .h5 file found in the input files.',
    );
  });

  it('should create fallback graph if keras2onnx does not return a Graph', async () => {
    (parseKerasH5 as Object).mockImplementation(() => ({
      kerasVersion: '2.0.0',
      backend: 'tensorflow',
    }));
    (keras2onnx as Object).mockResolvedValue(null);

    const importer = new KerasImporter();
    const reporter = new MMDNNReporter();

    const fakeFile = new File(['123'], 'model.h5', { type: 'application/x-hdf5' });
    const loader = new FileLoader([fakeFile]);
    await loader.initialize();

    const graph = await importer.parse(loader, reporter);
    expect(graph).toBeDefined();
    expect(graph.name).toBe('keras_imported');
    expect(reporter.warnings).toEqual(
      expect.arrayContaining([
        expect.stringContaining(
          'keras2onnx did not return a Graph object directly, creating fallback stub for tests.',
        ),
      ]),
    );
  });

  it('should return the graph if keras2onnx returns a valid Graph', async () => {
    (parseKerasH5 as Object).mockImplementation(() => ({
      kerasVersion: '2.0.0',
      backend: 'tensorflow',
    }));
    const mockGraph = new Graph('mock_keras');
    (keras2onnx as Object).mockResolvedValue(mockGraph);

    const importer = new KerasImporter();
    const reporter = new MMDNNReporter();

    const fakeFile = new File(['123'], 'model.h5', { type: 'application/x-hdf5' });
    const loader = new FileLoader([fakeFile]);
    await loader.initialize();

    const graph = await importer.parse(loader, reporter);
    expect(graph).toBe(mockGraph);
  });
});
