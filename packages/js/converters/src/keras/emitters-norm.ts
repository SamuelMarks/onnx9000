import { OnnxNodeBuilder } from './emitters.js';

/**
 * Emit a BatchNormalization node.
 */
export function emitBatchNormalization(
  inputName: string,
  outputName: string,
  gammaName: string,
  betaName: string,
  meanName: string,
  varName: string,
  epsilon: number,
  momentum: number,
  name: string,
): OnnxNodeBuilder[] {
  return [
    {
      opType: 'BatchNormalization',
      inputs: [inputName, gammaName, betaName, meanName, varName],
      outputs: [outputName],
      name,
      attributes: [
        { name: 'epsilon', f: epsilon, type: 'FLOAT' },
        { name: 'momentum', f: momentum, type: 'FLOAT' },
      ],
    },
  ];
}

/**
 * Emit a LayerNormalization node.
 */
export function emitLayerNormalization(
  inputName: string,
  outputName: string,
  gammaName: string | undefined,
  betaName: string | undefined,
  axis: number,
  epsilon: number,
  name: string,
): OnnxNodeBuilder[] {
  const inputs = [inputName];
  if (gammaName) inputs.push(gammaName);
  if (betaName) inputs.push(betaName);

  return [
    {
      opType: 'LayerNormalization',
      inputs,
      outputs: [outputName],
      name,
      attributes: [
        { name: 'axis', i: axis, type: 'INT' },
        { name: 'epsilon', f: epsilon, type: 'FLOAT' },
      ],
    },
  ];
}

/**
 * Emit a UnitNormalization (mapped to LpNormalization) node.
 */
export function emitUnitNormalization(
  inputName: string,
  outputName: string,
  axis: number,
  name: string,
): OnnxNodeBuilder[] {
  return [
    {
      opType: 'LpNormalization',
      inputs: [inputName],
      outputs: [outputName],
      name,
      attributes: [
        { name: 'axis', i: axis, type: 'INT' },
        { name: 'p', i: 2, type: 'INT' },
      ],
    },
  ];
}

/**
 * Emit a GroupNormalization node.
 */
export function emitGroupNormalization(
  inputName: string,
  outputName: string,
  numGroups: number,
  gammaName: string | undefined,
  betaName: string | undefined,
  epsilon: number,
  name: string,
): OnnxNodeBuilder[] {
  const inputs = [inputName];
  inputs.push(gammaName || '');
  inputs.push(betaName || '');

  return [
    {
      opType: 'GroupNormalization',
      inputs,
      outputs: [outputName],
      name,
      attributes: [
        { name: 'epsilon', f: epsilon, type: 'FLOAT' },
        { name: 'num_groups', i: numGroups, type: 'INT' },
      ],
    },
  ];
}

/**
 * Emit a Reshape node.
 */
export function emitReshape(
  inputName: string,
  shapeName: string,
  outputName: string,
  name: string,
): OnnxNodeBuilder[] {
  return [
    {
      opType: 'Reshape',
      inputs: [inputName, shapeName],
      outputs: [outputName],
      name,
      attributes: [],
    },
  ];
}

/**
 * Emit a Flatten node.
 */
export function emitFlatten(
  inputName: string,
  outputName: string,
  axis: number,
  name: string,
): OnnxNodeBuilder[] {
  return [
    {
      opType: 'Flatten',
      inputs: [inputName],
      outputs: [outputName],
      name,
      attributes: [{ name: 'axis', i: axis, type: 'INT' }],
    },
  ];
}

/**
 * Emit a Transpose node.
 */
export function emitTranspose(
  inputName: string,
  outputName: string,
  perm: number[],
  name: string,
): OnnxNodeBuilder[] {
  return [
    {
      opType: 'Transpose',
      inputs: [inputName],
      outputs: [outputName],
      name,
      attributes: [{ name: 'perm', ints: perm, type: 'INTS' }],
    },
  ];
}

/**
 * Emit a Pad node.
 */
export function emitPad(
  inputName: string,
  outputName: string,
  padsName: string,
  constantValueName: string | undefined,
  mode: 'constant' | 'reflect' | 'edge',
  name: string,
): OnnxNodeBuilder[] {
  const inputs = [inputName, padsName];
  if (constantValueName && mode === 'constant') {
    inputs.push(constantValueName);
  }

  return [
    {
      opType: 'Pad',
      inputs,
      outputs: [outputName],
      name,
      attributes: [{ name: 'mode', s: mode, type: 'STRING' }],
    },
  ];
}
