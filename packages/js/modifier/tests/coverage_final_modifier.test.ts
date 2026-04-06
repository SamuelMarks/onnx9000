import { describe, it, expect, vi } from 'vitest';
import { GraphMutator } from '../src/GraphMutator';
import { Graph, Node } from '@onnx9000/core';
import { ModelExporter } from '../src/components/export/exporter';

describe('Coverage Modifier', () => {
  it('GraphMutator strict mode rollback', () => {
    const graph = new Graph('test');
    const mutator = new GraphMutator(graph);
    mutator.strictMode = true;

    // Let's create a mutation that makes the graph invalid
    // A node with no inputs/outputs
    const badNode = new Node('Add', ['a', 'b'], ['c']);

    const mockMutation = {
      undo: vi.fn(),
      redo: vi.fn().mockImplementation(() => {
        graph.addNode(badNode);
      }),
    };

    expect(() => mutator.execute(mockMutation as Object)).toThrow('Strict Mode prevented');
    expect(mockMutation.undo).toHaveBeenCalled();

    // Also test empty undo/redo
    mutator.undo();
    mutator.redo();
  });

  it('Exporter save error', () => {
    const graph = new Graph('test');
    const mutator = new GraphMutator(graph);
    const exp = new ModelExporter(mutator);

    const alertMock = vi.fn();
    vi.stubGlobal('alert', alertMock);

    // Mock localStorage to throw
    const mockLs = {
      setItem: () => {
        throw new Error('Quota');
      },
    };
    vi.stubGlobal('localStorage', mockLs);

    exp.saveSessionToLocalStorage();
    expect(alertMock).toHaveBeenCalledWith('Failed to save session (might be too large).');

    vi.unstubAllGlobals();
  });
});
