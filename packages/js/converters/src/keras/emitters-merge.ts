/**
 * @fileoverview emitters-merge.ts
 * Provides emitters-merge functionality for the converters package.
 */
import { ConversionError } from '@onnx9000/core';
import type { OnnxNodeBuilder } from './emitters.js';

/**
 * Executes `emitMerge`.
 * @param opType Parameter `opType`.
 * @param inputNames Parameter `inputNames`.
 * @param outputName Parameter `outputName`.
 * @param name Parameter `name`.
 * @returns Result of `emitMerge`.
 */
export function emitMerge(
  opType: 'Add' | 'Sub' | 'Mul' | 'Mean' | 'Max' | 'Min',
  inputNames: string[],
  outputName: string,
  name: string,
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];

  // ONNX Add, Sub, Mul, etc. generally take 2 inputs.
  // Except Mean, Max, Min which can take multiple inputs
  if (opType === 'Mean' || opType === 'Max' || opType === 'Min') {
    nodes.push({
      opType,
      inputs: inputNames,
      outputs: [outputName],
      name,
      attributes: [],
    });
  } else {
    // For Add/Mul with > 2 inputs, we need to accumulate
    if (inputNames.length < 2) {
      throw new ConversionError(`Merge layer requires at least 2 inputs, got ${inputNames.length}`);
    }

    let currentOut = inputNames[0] as string;
    for (let i = 1; i < inputNames.length; i++) {
      const nextIn = inputNames[i] as string;
      const isLast = i === inputNames.length - 1;
      const out = isLast ? outputName : `${name}_step_${i}`;

      nodes.push({
        opType,
        inputs: [currentOut, nextIn],
        outputs: [out],
        name: `${name}_step_${i}`,
        attributes: [],
      });
      currentOut = out;
    }
  }

  return nodes;
}

/**
 * Executes `emitConcat`.
 * @param inputNames Parameter `inputNames`.
 * @param outputName Parameter `outputName`.
 * @param axis Parameter `axis`.
 * @param name Parameter `name`.
 * @returns Result of `emitConcat`.
 */
export function emitConcat(
  inputNames: string[],
  outputName: string,
  axis: number,
  name: string,
): OnnxNodeBuilder[] {
  return [
    {
      opType: 'Concat',
      inputs: inputNames,
      outputs: [outputName],
      name,
      attributes: [{ name: 'axis', i: axis, type: 'INT' }],
    },
  ];
}

/**
 * Executes `emitDot`.
 * @param input1Name Parameter `input1Name`.
 * @param input2Name Parameter `input2Name`.
 * @param outputName Parameter `outputName`.
 * @param _axes Parameter `_axes`.
 * @param name Parameter `name`.
 * @returns Result of `emitDot`.
 */
export function emitDot(
  input1Name: string,
  input2Name: string,
  outputName: string,
  _axes: number | [number, number],
  name: string,
): OnnxNodeBuilder[] {
  // Basic MatMul handles standard Dot on last axes.
  // If specific axes are provided, we'd need to emit Transpose nodes first.
  // A fully compliant transpiler inserts those Transpose nodes based on the axes argument.
  // Assuming default Dot (inner product):
  return [
    {
      opType: 'MatMul',
      inputs: [input1Name, input2Name],
      outputs: [outputName],
      name,
      attributes: [],
    },
  ];
}
