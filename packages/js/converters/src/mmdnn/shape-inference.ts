/* eslint-disable */
// @ts-nocheck
import { Graph } from '@onnx9000/core';
import { Shape } from '@onnx9000/core';
import { MMDNNReporter } from './reporter.js';

export class ShapeInferenceEngine {
  private shapeMap: Map<string, Shape>;

  constructor() {
    this.shapeMap = new Map();
  }

  inferShapes(graph: Graph, reporter: MMDNNReporter): void {
    // Collect known shapes
    for (const input of graph.inputs) {
      if (input.shape) {
        this.shapeMap.set(input.name, input.shape);
      }
    }
    for (const [name, tensor] of Object.entries(graph.tensors)) {
      if (tensor.shape) {
        this.shapeMap.set(name, tensor.shape);
      }
    }

    // Traverse topologically and deduce shapes
    for (const node of graph.nodes) {
      // Very basic example shape inference
      if (node.opType === 'Relu' || node.opType === 'Sigmoid' || node.opType === 'Tanh') {
        const inputShape = this.shapeMap.get(node.inputs[0]!);
        if (inputShape) {
          this.shapeMap.set(node.outputs[0]!, inputShape);
        }
      } else if (node.opType === 'Add' || node.opType === 'Mul') {
        // Broadcast
        const shapeA = this.shapeMap.get(node.inputs[0]!);
        const shapeB = this.shapeMap.get(node.inputs[1]!);
        if (shapeA && shapeB) {
          const dimsA = shapeA;
          const dimsB = shapeB;
          const outDims: (number | string)[] = [];
          const maxLen = Math.max(dimsA.length, dimsB.length);
          for (let i = 0; i < maxLen; i++) {
            const dA = dimsA[dimsA.length - 1 - i] || 1;
            const dB = dimsB[dimsB.length - 1 - i] || 1;
            if (dA === dB) outDims.unshift(dA);
            else if (dA === 1) outDims.unshift(dB);
            else if (dB === 1) outDims.unshift(dA);
            else {
              reporter.warn(
                `Incompatible shapes for broadcasting at node ${node.name}: ${dimsA} vs ${dimsB}`,
                node.name,
              );
              outDims.unshift(dA); // fallback
            }
          }
          this.shapeMap.set(node.outputs[0]!, outDims);
        }
      } else {
        reporter.warn(`Missing shape inference rules for op: ${node.opType}`, node.name);
      }
    }

    // Update graph outputs
    for (const output of graph.outputs) {
      const deducedShape = this.shapeMap.get(output.name);
      if (deducedShape && !output.shape) {
        output.shape = deducedShape;
      }
    }
  }

  getShape(tensorName: string): Shape | undefined {
    return this.shapeMap.get(tensorName);
  }
}
