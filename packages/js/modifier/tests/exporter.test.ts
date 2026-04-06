// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import { GraphMutator } from '../src/GraphMutator.js';
import { ModelExporter } from '../src/components/export/exporter.js';

describe('ModelExporter', () => {
  let graph: Graph;
  let mutator: GraphMutator;
  let exporter: ModelExporter;

  beforeEach(() => {
    graph = new Graph('Test');
    mutator = new GraphMutator(graph);
    exporter = new ModelExporter(mutator);

    global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    global.URL.revokeObjectURL = vi.fn();
  });

  it('101. 102. exportModel validates before export', async () => {
    // Valid empty graph
    const validBuf = await exporter.exportModel();
    expect(validBuf).toBeInstanceOf(Uint8Array);

    // Invalid graph (dangling node)
    mutator.addNode('Op', ['A'], ['B']);
    await expect(exporter.exportModel()).rejects.toThrow('Cannot export invalid graph');
  });

  it('103. downloadBlob triggers a click', () => {
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click');
    exporter.downloadBlob('test.bin', new Uint8Array([1, 2, 3]));
    expect(clickSpy).toHaveBeenCalled();
  });

  it('105. generateEditLog', () => {
    mutator.addInput('A', 'float32', [1]);
    const log = exporter.generateEditLog();
    expect(log).toContain('mutation_applied');
  });

  it('106. generatePythonHelperScript', () => {
    mutator.addNode('Relu', ['X'], ['Y'], {}, 'MyRelu');
    const py = exporter.generatePythonHelperScript();
    expect(py).toContain('import onnx');
    expect(py).toContain("helper.make_node('Relu'");
    expect(py).toContain('["X"]');
    expect(py).toContain('["Y"]');
  });

  it('107. generateSummary computes params', () => {
    mutator.addInitializer('W1', 'float32', [2, 2], new Float32Array([1, 2, 3, 4]));
    const summary = exporter.generateSummary();
    expect(summary).toContain('Nodes: 0');
    expect(summary).toContain('Initializers: 1');
    expect(summary).toContain('Parameters: 4');

    mutator.graph.initializers.push('BadInit');
    expect(exporter.generateSummary()).toContain('Parameters: 4'); // graceful skip
  });

  it('109. generateGraphvizDot', () => {
    mutator.addInput('A', 'float32', [1]);
    mutator.addNode('Relu', ['A'], ['B'], {}, 'MyRelu');
    const dot = exporter.generateGraphvizDot();
    expect(dot).toContain('digraph G {');
    expect(dot).toContain('shape=oval');
    expect(dot).toContain('shape=box');
    expect(dot).toContain('->');
  });

  it('handles unnamed nodes for code gen', () => {
    mutator.addNode('Relu', ['X'], ['Y'], {});
    const py = exporter.generatePythonHelperScript();
    const dot = exporter.generateGraphvizDot();
    expect(py).toContain('helper.make_node');
    expect(dot).toContain('shape=box');
  });

  it('206. exportModel enforces memory thresholds', async () => {
    // Add large tensor
    const data = new Float32Array((1001 * 1024 * 1024) / 4); // ~1GB
    mutator.addInitializer('W', 'float32', [data.length], data);

    // Mock confirm to return false
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);

    await expect(exporter.exportModel()).rejects.toThrow('memory threshold warning');
    expect(confirmSpy).toHaveBeenCalled();
  });

  it('220. exportStatsCSV downloads blob', () => {
    mutator.addNode('Conv', [], []);
    mutator.addNode('Conv', [], []);
    mutator.addNode('Relu', [], []);

    const createSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:url');
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    exporter.exportStatsCSV();
    expect(createSpy).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalled();
  });

  it('239. saveSessionToLocalStorage stores graph summary', () => {
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem');
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});

    exporter.saveSessionToLocalStorage();
    expect(setItemSpy).toHaveBeenCalledWith('onnx_modifier_session_graph', expect.any(String));
    expect(alertSpy).toHaveBeenCalledWith('Session state saved locally.');
  });

  it('290. promptChangesBeforeExport shows summary of edits', () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    mutator.addNode('Conv', [], []);
    // simulate edit count
    (mutator as Object).deletedNodeCount = 5;

    const res = exporter.promptChangesBeforeExport();
    expect(res).toBe(true);
    expect(confirmSpy).toHaveBeenCalledWith(
      expect.stringContaining('Nodes deleted this session: 5'),
    );
  });
});
