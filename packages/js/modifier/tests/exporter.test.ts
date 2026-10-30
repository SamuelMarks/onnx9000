import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { describe, expect, it, vi } from 'vitest';
import { ModelExporter } from '../src/components/export/exporter.js';
import { GraphMutator } from '../src/GraphMutator.js';

describe('ModelExporter', () => {
  it('should export dot and csv', () => {
    const g = new Graph('test');
    g.inputs.push(new ValueInfo('in', [1], 'float32'));
    g.nodes.push(new Node('Add', ['in'], ['out'], {}, 'n1'));
    const mutator = new GraphMutator(g);
    const exporter = new ModelExporter(mutator);

    expect(exporter.generateGraphvizDot()).toContain('digraph G');

    // mock URL for CSV export
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob');
    global.URL.revokeObjectURL = vi.fn();
    exporter.exportStatsCSV();
    expect(global.URL.createObjectURL).toHaveBeenCalled();
  });
});
