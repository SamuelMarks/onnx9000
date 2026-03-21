import { Graph } from '@onnx9000/core';
import { MMDNNReporter } from './reporter.js';

export type Layout = 'NCHW' | 'NHWC' | 'UNKNOWN';

export class DataLayoutTracker {
  private tensorLayouts: Map<string, Layout>;

  constructor() {
    this.tensorLayouts = new Map();
  }

  track(graph: Graph, reporter: MMDNNReporter): void {
    // Assume all image inputs start as NCHW unless otherwise known
    for (const input of graph.inputs) {
      if (input.shape && input.shape.length === 4) {
        this.tensorLayouts.set(input.name, 'NCHW');
        reporter.info(`Assuming NCHW for 4D input ${input.name}`);
      }
    }

    for (const node of graph.nodes) {
      if (node.inputs.length === 0) continue;
      const primaryInput = node.inputs[0]!;
      const layout = this.tensorLayouts.get(primaryInput) || 'UNKNOWN';

      if (node.opType === 'Transpose') {
        // Simple heuristic: if transposing NCHW (0,1,2,3) to NHWC (0,2,3,1)
        const perm = node.attributes['perm'];
        if (perm && Array.isArray(perm.value) && perm.value.length === 4) {
          const p = perm.value as number[];
          if (p[0] === 0 && p[1] === 2 && p[2] === 3 && p[3] === 1) {
            this.tensorLayouts.set(node.outputs[0]!, 'NHWC');
            continue;
          } else if (p[0] === 0 && p[1] === 3 && p[2] === 1 && p[3] === 2) {
            this.tensorLayouts.set(node.outputs[0]!, 'NCHW');
            continue;
          }
        }
      }

      // Propagate layout
      for (const output of node.outputs) {
        this.tensorLayouts.set(output, layout);
      }
    }
  }

  getLayout(tensorName: string): Layout {
    return this.tensorLayouts.get(tensorName) || 'UNKNOWN';
  }
}
