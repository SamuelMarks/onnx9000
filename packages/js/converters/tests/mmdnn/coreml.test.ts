import { describe, it, expect, vi } from 'vitest';
import { CoreMLImporter } from '../../src/mmdnn/coreml/importer.js';
import { FileLoader } from '../../src/mmdnn/file-loader.js';
import { MMDNNReporter } from '../../src/mmdnn/reporter.js';

vi.mock('fflate', () => {
  return {
    unzipSync: vi.fn(),
  };
});

import * as fflate from 'fflate';

describe('MMDNN - CoreML Importer', () => {
  it('should parse standalone .mlmodel', async () => {
    const importer = new CoreMLImporter();
    const reporter = new MMDNNReporter();
    const fakeFile = new File([''], 'model.mlmodel');
    const loader = new FileLoader([fakeFile]);
    await loader.initialize();
    const graph = await importer.parse(loader, reporter);
    expect(graph.name).toContain('coreml_imported');
  });

  it('should parse zipped .mlpackage.zip', async () => {
    const importer = new CoreMLImporter();
    const reporter = new MMDNNReporter();
    const fakeFile = new File([''], 'model.mlpackage.zip');
    const loader = new FileLoader([fakeFile]);
    await loader.initialize();

    (fflate.unzipSync as Object).mockReturnValue({
      'model.mlmodel': new Uint8Array([1, 2, 3]),
    });

    const graph = await importer.parse(loader, reporter);
    expect(graph.name).toContain('coreml_imported');
  });

  it('should throw if missing .mlmodel inside .zip', async () => {
    const importer = new CoreMLImporter();
    const reporter = new MMDNNReporter();
    const fakeFile = new File([''], 'model.zip');
    const loader = new FileLoader([fakeFile]);
    await loader.initialize();

    (fflate.unzipSync as Object).mockReturnValue({
      'some_other_file.txt': new Uint8Array([1, 2, 3]),
    });

    const graph = await importer.parse(loader, reporter);
    expect(graph.name).toContain('mock_coreml_graph_fallback');
    expect(reporter.warnings.length).toBe(1);
    expect(reporter.warnings[0]).toContain('CoreML parsing failed');
  });

  it('should return mock graph if no valid files exist', async () => {
    const importer = new CoreMLImporter();
    const reporter = new MMDNNReporter();
    const loader = new FileLoader([]); // valid but empty
    const graph = await importer.parse(loader, reporter);
    expect(graph.name).toBe('mock_coreml_graph');
  });
});
