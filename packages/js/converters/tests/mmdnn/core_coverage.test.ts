import { describe, it, expect, vi } from 'vitest';
import { MMDNNReporter, MMDNNError } from '../../src/mmdnn/reporter.js';
import { FileLoader } from '../../src/mmdnn/file-loader.js';
import { topologicalSort } from '../../src/mmdnn/topology.js';
import { NodeFusionRegistry } from '../../src/mmdnn/fusion.js';
import { DataLayoutTracker } from '../../src/mmdnn/layout.js';
import { ShapeInferenceEngine } from '../../src/mmdnn/shape-inference.js';
import { Graph, Node, ValueInfo, Tensor, Shape, Attribute } from '@onnx9000/core';

describe('MMDNN - Core Files Full Coverage', () => {
  describe('MMDNNReporter', () => {
    it('should log with verbose=true', () => {
      const reporter = new MMDNNReporter(true);
      const consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      reporter.info('test info');
      expect(consoleLogSpy).toHaveBeenCalledWith('[INFO] test info');

      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      reporter.warn('test warn');
      expect(consoleWarnSpy).toHaveBeenCalledWith('[WARN]: test warn');

      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const consoleTraceSpy = vi.spyOn(console, 'trace').mockImplementation(() => {});
      expect(() => reporter.error('test error', 'node1')).toThrow(MMDNNError);
      expect(consoleErrorSpy).toHaveBeenCalledWith('[ERROR] (Node: node1): test error');
      expect(consoleTraceSpy).toHaveBeenCalled();

      const report = reporter.getReport();
      expect(report).toContain('--- MMDNN Conversion Report ---');
      expect(report).toContain('[WARN]: test warn');
      expect(report).toContain('[ERROR] (Node: node1): test error');

      vi.restoreAllMocks();
    });
  });

  describe('FileLoader', () => {
    it('should throw on unsupported extensions', () => {
      const fakeFile = new File([''], 'test.unsupported', { type: 'text/plain' });
      expect(() => new FileLoader([fakeFile])).toThrow('Unsupported file type');
    });

    it('should initialize and read slices', async () => {
      const fakeBlob = new Blob(['1234567890']);
      const loader = new FileLoader([fakeBlob]);
      await loader.initialize(); // Coverage

      // Blob has no filename so it generates a blob_... name
      const keys = Array.from((loader as any).files.keys());
      expect(keys[0]).toContain('blob_');

      const sliceBuffer = await loader.readSlice(keys[0], 0, 5);
      expect(sliceBuffer.byteLength).toBe(5);

      const sliceText = await loader.readSliceText(keys[0], 5, 10);
      expect(sliceText).toBe('67890');
    });

    it('should throw when getting non-existent file', () => {
      const loader = new FileLoader([]);
      expect(() => loader.getFile('missing')).toThrow('File not found: missing');
    });
  });

  describe('Topology Sorter', () => {
    it('should handle explicit tensors and report missing inputs/cycles', () => {
      const graph = new Graph('test');
      graph.tensors['my_tensor'] = new Tensor('my_tensor', [1], 'float32', true);

      const nodeA = new Node('Relu', ['input0'], ['A_out'], {}, 'NodeA');
      const nodeB = new Node('Relu', ['A_out'], ['B_out'], {}, 'NodeB');

      // Create cycle
      const nodeC = new Node('Relu', ['B_out'], ['input0'], {}, 'NodeC');

      graph.nodes = [nodeA, nodeB, nodeC];

      const reporter = new MMDNNReporter(false);
      expect(() => topologicalSort(graph, reporter)).toThrow('Cyclic graph detected');
    });

    it('should warn on missing tensor without throwing if no cycle', () => {
      const graph = new Graph('test');
      const nodeA = new Node('Relu', ['missing_input'], ['A_out'], {}, 'NodeA');
      graph.nodes = [nodeA];
      const reporter = new MMDNNReporter(false);
      topologicalSort(graph, reporter);
      expect(reporter.warnings.length).toBe(1);
      expect(reporter.warnings[0]).toContain('missing_input');
    });
  });

  describe('NodeFusionRegistry', () => {
    it('should fuse Conv + BatchNorm', () => {
      const graph = new Graph('fusion');
      const conv = new Node('Conv', ['x', 'w'], ['conv_out'], {}, 'conv1');
      const bn = new Node(
        'BatchNormalization',
        ['conv_out', 'scale', 'b', 'mean', 'var'],
        ['bn_out'],
        {},
        'bn1',
      );
      graph.nodes = [conv, bn];

      const registry = new NodeFusionRegistry();
      const reporter = new MMDNNReporter(false);
      const fused = registry.applyFusions(graph, reporter);

      expect(fused.nodes.length).toBe(1);
      expect(fused.nodes[0].opType).toBe('Conv');
      expect(fused.nodes[0].outputs).toEqual(['bn_out']);
    });

    it('should not fuse if Conv output is used multiple times', () => {
      const graph = new Graph('fusion');
      const conv = new Node('Conv', ['x', 'w'], ['conv_out'], {}, 'conv1');
      const bn = new Node(
        'BatchNormalization',
        ['conv_out', 'scale', 'b', 'mean', 'var'],
        ['bn_out'],
        {},
        'bn1',
      );
      const other = new Node('Relu', ['conv_out'], ['relu_out'], {}, 'relu1');
      graph.nodes = [conv, bn, other];

      const registry = new NodeFusionRegistry();
      const reporter = new MMDNNReporter(false);
      const fused = registry.applyFusions(graph, reporter);
      expect(fused.nodes.length).toBe(3); // Unfused
    });

    it('should fuse Add + MatMul -> Gemm (producerB)', () => {
      const graph = new Graph('fusion');
      const matmul = new Node('MatMul', ['w', 'x'], ['matmul_out'], {}, 'mm1');
      // MatMul is the 2nd input to Add
      const add = new Node('Add', ['b', 'matmul_out'], ['add_out'], {}, 'add1');
      graph.nodes = [matmul, add];

      const registry = new NodeFusionRegistry();
      const reporter = new MMDNNReporter(false);
      const fused = registry.applyFusions(graph, reporter);

      expect(fused.nodes.length).toBe(1);
      expect(fused.nodes[0].opType).toBe('Gemm');
      expect(fused.nodes[0].inputs).toEqual(['w', 'x', 'b']);
    });

    it('should not fuse Add if not MatMul', () => {
      const graph = new Graph('fusion');
      const conv = new Node('Conv', ['x', 'w'], ['conv_out'], {}, 'conv1');
      const add = new Node('Add', ['conv_out', 'b'], ['add_out'], {}, 'add1');
      graph.nodes = [conv, add];

      const registry = new NodeFusionRegistry();
      const reporter = new MMDNNReporter(false);
      const fused = registry.applyFusions(graph, reporter);

      expect(fused.nodes.length).toBe(2);
    });

    it('should not fuse Add + MatMul if MatMul output is used multiple times', () => {
      const graph = new Graph('fusion');
      const matmul = new Node('MatMul', ['w', 'x'], ['matmul_out'], {}, 'mm1');
      const add = new Node('Add', ['b', 'matmul_out'], ['add_out'], {}, 'add1');
      const other = new Node('Relu', ['matmul_out'], ['relu_out'], {}, 'relu1');
      graph.nodes = [matmul, add, other];

      const registry = new NodeFusionRegistry();
      const reporter = new MMDNNReporter(false);
      const fused = registry.applyFusions(graph, reporter);

      expect(fused.nodes.length).toBe(3);
    });
  });

  describe('DataLayoutTracker', () => {
    it('should handle zero inputs and UNKNOWN fallback', () => {
      const graph = new Graph('layout_test');
      const constant = new Node('Constant', [], ['const_out'], {}, 'const1');
      graph.nodes = [constant];
      const tracker = new DataLayoutTracker();
      const reporter = new MMDNNReporter(false);
      tracker.track(graph, reporter);
      expect(tracker.getLayout('const_out')).toBe('UNKNOWN');
      expect(tracker.getLayout('missing')).toBe('UNKNOWN');
    });
    it('should track NHWC back to NCHW', () => {
      const graph = new Graph('layout_test');
      const transpose = new Node(
        'Transpose',
        ['image_nhwc'],
        ['image_nchw'],
        {
          perm: new Attribute('perm', 'INTS', [0, 3, 1, 2]),
        },
        'transpose2',
      );
      graph.nodes = [transpose];

      const tracker = new DataLayoutTracker();
      const reporter = new MMDNNReporter(false);
      tracker['tensorLayouts'].set('image_nhwc', 'NHWC');
      tracker.track(graph, reporter);

      expect(tracker.getLayout('image_nchw')).toBe('NCHW');
    });
  });

  describe('ShapeInferenceEngine', () => {
    it('should warn on incompatible broadcasting shapes', () => {
      const graph = new Graph('shape_test');
      graph.tensors['A'] = new Tensor('A', [1, 3, 224, 224], 'float32', true);
      graph.tensors['B'] = new Tensor('B', [1, 5, 224, 224], 'float32', true);
      const add = new Node('Add', ['A', 'B'], ['C'], {}, 'AddNode');
      graph.nodes = [add];

      const reporter = new MMDNNReporter(false);
      const engine = new ShapeInferenceEngine();
      engine.inferShapes(graph, reporter);

      expect(reporter.warnings.length).toBeGreaterThan(0);
      expect(reporter.warnings[0]).toContain('Incompatible shapes for broadcasting');
    });

    it('should warn on missing shape inference rules', () => {
      const graph = new Graph('shape_test');
      graph.tensors['A'] = new Tensor('A', [1, 3], 'float32', true);
      const unknown = new Node('UnknownOp', ['A'], ['B'], {}, 'UnknownNode');
      graph.nodes = [unknown];

      const reporter = new MMDNNReporter(false);
      const engine = new ShapeInferenceEngine();
      engine.inferShapes(graph, reporter);

      expect(reporter.warnings.length).toBeGreaterThan(0);
      expect(reporter.warnings[0]).toContain('Missing shape inference rules for op: UnknownOp');
    });

    it('should deduce and set graph outputs shape if missing', () => {
      const graph = new Graph('shape_test');
      graph.tensors['A'] = new Tensor('A', [1, 3], 'float32', true);
      const relu = new Node('Relu', ['A'], ['B'], {}, 'ReluNode');
      graph.nodes = [relu];

      const outVI = new ValueInfo('B', [], 'float32');
      outVI.shape = undefined as any;
      graph.outputs = [outVI];

      const reporter = new MMDNNReporter(false);
      const engine = new ShapeInferenceEngine();
      engine.inferShapes(graph, reporter);

      expect(graph.outputs[0].shape).toBeDefined();
      expect(graph.outputs[0].shape).toEqual([1, 3]);
    });
  });
});
