import { Graph, Node } from '@onnx9000/core';
import { Block, Region, Operation, Value } from '../ir/core.js';
import { TensorType } from '../dialects/web/tensor.js';
import * as mhlo from '../dialects/web/mhlo.js';

function getONNXType(graph: Graph, name: string): TensorType {
  const v = graph.inputs.find((x) => x.name === name) || graph.outputs.find((x) => x.name === name);
  if (v) {
    const shape = v.shape.map((s) => (typeof s === 'number' ? s : -1));
    return new TensorType(shape, v.dtype);
  }
  const t = graph.tensors[name];
  if (t) {
    const shape = t.shape.map((s) => (typeof s === 'number' ? s : -1));
    return new TensorType(shape, t.dtype);
  }
  return new TensorType([], 'unknown');
}

export function lowerONNXToMHLO(onnxGraph: Graph): Region {
  const region = new Region();
  const block = new Block(region);
  region.pushBlock(block);

  const valueMap = new Map<string, Value>();

  for (const input of onnxGraph.inputs) {
    const type = new TensorType(
      input.shape.map((s) => (typeof s === 'number' ? s : -1)),
      input.dtype,
    );
    const arg = block.addArgument(type);
    valueMap.set(input.name, arg);
  }

  for (const initName of onnxGraph.initializers) {
    const t = onnxGraph.tensors[initName];
    if (t) {
      const shape = t.shape.map((s) => (typeof s === 'number' ? s : -1));
      const type = new TensorType(shape, t.dtype);
      const constOp = new Operation('web.mhlo.constant', [], [type], { value: t });
      block.pushOperation(constOp);
      valueMap.set(initName, constOp.results[0]!);
    }
  }

  for (const node of onnxGraph.nodes) {
    const getOperand = (name: string | undefined) => {
      if (!name) throw new Error(`Operand missing`);
      const v = valueMap.get(name);
      if (!v) {
        throw new Error(`Operand ${name} not found`);
      }
      return v;
    };

    const resultTypes = node.outputs.map((out) => getONNXType(onnxGraph, out!));
    const resType = resultTypes[0]!;

    let op: Operation;

    switch (node.opType) {
      case 'Add':
        op = mhlo.add(getOperand(node.inputs[0]), getOperand(node.inputs[1]), resType);
        break;
      case 'Sub':
        op = mhlo.subtract(getOperand(node.inputs[0]), getOperand(node.inputs[1]), resType);
        break;
      case 'Mul':
        op = mhlo.multiply(getOperand(node.inputs[0]), getOperand(node.inputs[1]), resType);
        break;
      case 'Div':
        op = mhlo.divide(getOperand(node.inputs[0]), getOperand(node.inputs[1]), resType);
        break;
      case 'MatMul':
        op = mhlo.dot(getOperand(node.inputs[0]), getOperand(node.inputs[1]), resType);
        break;
      case 'Exp':
        op = mhlo.exponential(getOperand(node.inputs[0]), resType);
        break;
      case 'Log':
        op = mhlo.log(getOperand(node.inputs[0]), resType);
        break;
      case 'Cos':
        op = mhlo.cosine(getOperand(node.inputs[0]), resType);
        break;
      case 'Sin':
        op = mhlo.sine(getOperand(node.inputs[0]), resType);
        break;
      case 'Max':
        op = mhlo.maximum(getOperand(node.inputs[0]), getOperand(node.inputs[1]), resType);
        break;
      case 'Min':
        op = mhlo.minimum(getOperand(node.inputs[0]), getOperand(node.inputs[1]), resType);
        break;
      case 'Conv': {
        const strides = (node.attributes['strides']?.value as number[]) || [1, 1];
        const pads = (node.attributes['pads']?.value as number[]) || [0, 0, 0, 0];
        const dilations = (node.attributes['dilations']?.value as number[]) || [1, 1];
        const mhloPadding: number[][] = [];
        const spatialDims = strides.length;
        for (let i = 0; i < spatialDims; i++) {
          const low = pads[i] || 0;
          const high = pads[i + spatialDims] || 0;
          mhloPadding.push([low, high]);
        }

        op = mhlo.convolution(
          getOperand(node.inputs[0]),
          getOperand(node.inputs[1]),
          strides,
          mhloPadding,
          [1, 1],
          dilations,
          strides.map(() => false),
          resType,
        );
        break;
      }
      case 'Reshape': {
        op = mhlo.reshape(getOperand(node.inputs[0]), resType);
        break;
      }
      case 'Transpose': {
        const perm = (node.attributes['perm']?.value as number[]) || [];
        op = mhlo.transpose(getOperand(node.inputs[0]), perm, resType);
        break;
      }
      case 'Concat': {
        const axis = (node.attributes['axis']?.value as number) || 0;
        op = mhlo.concatenate(node.inputs.map(getOperand), axis, resType);
        break;
      }
      case 'Slice': {
        op = mhlo.dynamicSlice(
          getOperand(node.inputs[0]),
          [getOperand(node.inputs[1])],
          (resType as TensorType).shape,
          resType,
        );
        break;
      }
      default:
        op = new Operation('web.mhlo.custom_call', node.inputs.map(getOperand), resultTypes, {
          call_target_name: node.opType,
        });
        break;
    }

    block.pushOperation(op);

    for (let i = 0; i < node.outputs.length; i++) {
      valueMap.set(node.outputs[i]!, op.results[i]!);
    }
  }

  const returnOp = new Operation(
    'web.mhlo.return',
    onnxGraph.outputs.map((out) => {
      const v = valueMap.get(out.name);
      if (!v) {
        throw new Error(`Output ${out.name} not found`);
      }
      return v;
    }),
    [],
  );
  block.pushOperation(returnOp);

  return region;
}
