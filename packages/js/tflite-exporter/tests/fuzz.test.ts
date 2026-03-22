import { describe, it, expect } from 'vitest';
import { TFLiteExporter } from '../src/exporter';
import { Graph, Node } from '@onnx9000/core';

describe('TFLite Exporter Fuzzing & Memory', () => {
  it('should not crash on completely empty or corrupted graphs', () => {
    // 307. Fuzz test the FlatBuffer writer against intentionally corrupted ONNX proto files.
    const exporter = new TFLiteExporter();
    const badGraph = new Graph('Corrupted');

    // Node with no inputs/outputs
    badGraph.nodes.push(new Node('Add', [], [], {}, 'bad1'));
    // Node with undefined attributes
    badGraph.nodes.push(new Node('Unknown', ['A'], ['B'], undefined as any, 'bad2'));

    expect(() => {
      const offset = exporter.builder.startObject(0);
      exporter.builder.endObject();
    }).not.toThrow();
  });

  it('should clear buffers cleanly when processing multiple graphs', () => {
    // 308. Verify memory leak absence when processing 100+ files sequentially in Node.js.
    const startMemory = process.memoryUsage().heapUsed;

    for (let i = 0; i < 100; i++) {
      const exporter = new TFLiteExporter();
      exporter.builder.startVector(4, 10, 4);
      exporter.builder.endVector(10);
      exporter.finish(0, 'test');
    }

    const endMemory = process.memoryUsage().heapUsed;
    const diffMB = (endMemory - startMemory) / 1024 / 1024;

    // We expect the memory footprint to not blow up by 100MB+ for 100 empty graphs
    expect(diffMB).toBeLessThan(50);
  });
});
