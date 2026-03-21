import { OnnxNodeBuilder } from './emitters.js';

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

export function emitReshape(
  inputName: string,
  shapeName: string, // Tensor containing target shape
  outputName: string,
  name: string,
): OnnxNodeBuilder[] {
  return [
    {
      opType: 'Reshape',
      inputs: [inputName, shapeName],
      outputs: [outputName],
      name,
      attributes: [], // ONNX 14 allows 'allowzero' attribute, omitted for simplicity
    },
  ];
}

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

export function emitPad(
  inputName: string,
  outputName: string,
  padsName: string, // Tensor containing paddings [x1_begin, x2_begin... x1_end, x2_end...]
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
