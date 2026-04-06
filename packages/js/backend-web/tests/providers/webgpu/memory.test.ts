import { describe, it, expect } from 'vitest';
import { WebGPUMemoryManager } from '../../../src/providers/webgpu/memory';
import { Graph, Node } from '@onnx9000/core';

describe('WebGPUMemoryManager', () => {
  it('should partition graph correctly when memory limit is exceeded', () => {
    const manager = new WebGPUMemoryManager(25 * 1024 * 1024); // 25MB limit

    const g = new Graph('test');
    // Add 5 nodes, each is 10MB
    for (let i = 0; i < 5; i++) {
      g.nodes.push(new Node('DummyOp'));
    }

    const partitions = manager.checkAndPartition(g);

    // 50MB total, 25MB limit = 3 partitions (20MB, 20MB, 10MB)
    expect(partitions.length).toBe(3);
    expect(partitions[0].nodes.length).toBe(2);
    expect(partitions[1].nodes.length).toBe(2);
    expect(partitions[2].nodes.length).toBe(1);
  });
});
