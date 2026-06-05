import { describe, it, expect, vi } from 'vitest';
import { ModelExporter } from '../src/components/export/exporter.js';

describe('ModelExporter', () => {
  it('should export dot and csv', () => {
    const mutator: any = {
      graph: {
        inputs: [{ name: 'in' }],
        outputs: [],
        nodes: [{ id: 'n1', opType: 'Add', name: 'n1', inputs: ['in'], outputs: ['out'] }],
        initializers: [],
        tensors: {},
      },
    };
    const exporter = new ModelExporter(mutator);

    expect(exporter.generateGraphvizDot()).toContain('digraph G');

    // mock URL for CSV export
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob');
    global.URL.revokeObjectURL = vi.fn();
    exporter.exportStatsCSV();
    expect(global.URL.createObjectURL).toHaveBeenCalled();
  });
});
