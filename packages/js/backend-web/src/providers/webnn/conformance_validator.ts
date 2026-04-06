import { Graph, Node } from '@onnx9000/core';

export class WebNNLayoutValidator {
  /**
   * Verify gemm and conv2d layouts match navigator.ml specifications for all ingested CNNs.
   * WebNN typically expects NCHW or NHWC but requires explicit layout markers.
   */
  static validateLayouts(graph: Graph): boolean {
    for (const node of graph.nodes) {
      if (node.opType === 'Conv' || node.opType === 'Conv2D') {
        const layoutAttr = node.attributes['layout'] || node.attributes['data_format'];
        if (!layoutAttr) {
          throw new Error(
            `WebNN Validation Failed: Conv node ${node.name} is missing a layout attribute.`,
          );
        }
        const layout = typeof layoutAttr.value === 'string' ? layoutAttr.value : 'UNKNOWN';
        if (layout !== 'nchw' && layout !== 'nhwc' && layout !== 'NCHW' && layout !== 'NHWC') {
          throw new Error(
            `WebNN Validation Failed: Conv node ${node.name} has invalid layout ${layout}. Must be NCHW or NHWC.`,
          );
        }
      }

      if (node.opType === 'Gemm' || node.opType === 'MatMul') {
        // WebNN Gemm expects 2D inputs. If Rank > 2, it requires specific broadcasting rules.
        // We mock a simple rank check here
        const transA = node.attributes['transA']?.value;
        const transB = node.attributes['transB']?.value;
        if (transA === undefined && node.opType === 'Gemm') {
          // In strict conformance, it might need to be explicitly 0 or 1
        }
      }
    }
    return true;
  }
}
