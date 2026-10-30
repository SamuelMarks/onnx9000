/**
 * @fileoverview emitters.ts
 * Provides emitters functionality for the converters package.
 */
import { ConversionError } from '@onnx9000/core';
/** Interface `OnnxNodeBuilder` definition. */
export interface OnnxNodeBuilder {
  /** Property `opType`. */
  opType: string;
  /** Property `inputs`. */
  inputs: string[];
  /** Property `outputs`. */
  outputs: string[];
  /** Property `name`. */
  name: string;
  /** Property `attributes`. */
  attributes: {
    name: string;
    type?: string;
    f?: number;
    i?: number;
    ints?: number[];
    floats?: number[];
    s?: string;
  }[];
}

/** Interface `ActivationOptions` definition. */
export interface ActivationOptions {
  /** Property `alpha`. */
  alpha?: number;
  /** Property `theta`. */
  theta?: number;
  /** Property `alphaWeightName`. */
  alphaWeightName?: string;
}

/**
 * Executes `emitActivation`.
 * @param activation Parameter `activation`.
 * @param inputName Parameter `inputName`.
 * @param outputName Parameter `outputName`.
 * @param name Parameter `name`.
 * @param options Parameter `options`.
 * @returns Result of `emitActivation`.
 */
export function emitActivation(
  activation: string,
  inputName: string,
  outputName: string,
  name: string,
  options?: ActivationOptions,
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];
  switch (activation) {
    case 'relu':
      nodes.push({
        opType: 'Relu',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [],
      });
      break;
    case 'softmax':
      nodes.push({
        opType: 'Softmax',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [{ name: 'axis', i: -1, type: 'INT' }],
      });
      break;
    case 'sigmoid':
      nodes.push({
        opType: 'Sigmoid',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [],
      });
      break;
    case 'tanh':
      nodes.push({
        opType: 'Tanh',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [],
      });
      break;
    case 'softplus':
      nodes.push({
        opType: 'Softplus',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [],
      });
      break;
    case 'softsign':
      nodes.push({
        opType: 'Softsign',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [],
      });
      break;
    case 'linear':
      nodes.push({
        opType: 'Identity',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [],
      });
      break;
    case 'elu':
      nodes.push({
        opType: 'Elu',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [{ name: 'alpha', f: options?.alpha || 1.0, type: 'FLOAT' }],
      });
      break;
    case 'selu':
      nodes.push({
        opType: 'Selu',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [
          {
            name: 'alpha',
            f: 1.6732632423543772,
            type: 'FLOAT',
          },
          {
            name: 'gamma',
            f: 1.0507009873554805,
            type: 'FLOAT',
          },
        ],
      });
      break;
    case 'leaky_relu':
      nodes.push({
        opType: 'LeakyRelu',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [{ name: 'alpha', f: options?.alpha || 0.3, type: 'FLOAT' }],
      });
      break;
    case 'prelu':
      // PReLU requires a learnable parameter 'slope' passed as an input. Assume it is options.alphaWeightName.
      nodes.push({
        opType: 'PRelu',
        inputs: [inputName, options?.alphaWeightName || ''],
        outputs: [outputName],
        name,
        attributes: [],
      });
      break;
    case 'thresholded_relu':
      nodes.push({
        opType: 'ThresholdedRelu',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [{ name: 'alpha', f: options?.theta || 1.0, type: 'FLOAT' }],
      });
      break;
    case 'swish':
    case 'silu':
      nodes.push({
        opType: 'Sigmoid',
        inputs: [inputName],
        outputs: [`${name}_sig`],
        name: `${name}_sigmoid`,
        attributes: [],
      });
      nodes.push({
        opType: 'Mul',
        inputs: [inputName, `${name}_sig`],
        outputs: [outputName],
        name: `${name}_mul`,
        attributes: [],
      });
      break;
    case 'hard_swish':
      nodes.push({
        opType: 'HardSwish',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [],
      });
      break;
    case 'mish':
      nodes.push({
        opType: 'Mish',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [],
      });
      break;
    case 'gelu':
      nodes.push({
        opType: 'Gelu',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [{ name: 'approximate', s: 'none', type: 'STRING' }],
      });
      break;
    case 'hard_sigmoid':
      nodes.push({
        opType: 'HardSigmoid',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [
          { name: 'alpha', f: 0.2, type: 'FLOAT' },
          { name: 'beta', f: 0.5, type: 'FLOAT' },
        ],
      });
      break;
    default:
      throw new ConversionError(`Unsupported activation: ${activation}`);
  }
  return nodes;
}

/**
 * Executes `emitDense`.
 * @param inputName Parameter `inputName`.
 * @param outputName Parameter `outputName`.
 * @param weightName Parameter `weightName`.
 * @param biasName Parameter `biasName`.
 * @param activation Parameter `activation`.
 * @param name Parameter `name`.
 * @returns Result of `emitDense`.
 */
export function emitDense(
  inputName: string,
  outputName: string,
  weightName: string,
  biasName: string | undefined,
  activation: string,
  name: string,
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];
  const matmulOut = biasName || activation !== 'linear' ? `${name}_matmul` : outputName;

  nodes.push({
    opType: 'MatMul',
    inputs: [inputName, weightName],
    outputs: [matmulOut],
    name: `${name}_matmul`,
    attributes: [],
  });

  let addOut = matmulOut;
  if (biasName) {
    addOut = activation !== 'linear' ? `${name}_add` : outputName;
    nodes.push({
      opType: 'Add',
      inputs: [matmulOut, biasName],
      outputs: [addOut],
      name: `${name}_add`,
      attributes: [],
    });
  }

  if (activation && activation !== 'linear') {
    nodes.push(...emitActivation(activation, addOut, outputName, `${name}_act`));
  }

  return nodes;
}

/**
 * Executes `emitIdentity`.
 * @param inputName Parameter `inputName`.
 * @param outputName Parameter `outputName`.
 * @param name Parameter `name`.
 * @returns Result of `emitIdentity`.
 */
export function emitIdentity(
  inputName: string,
  outputName: string,
  name: string,
): OnnxNodeBuilder[] {
  return [
    {
      opType: 'Identity',
      inputs: [inputName],
      outputs: [outputName],
      name,
      attributes: [],
    },
  ];
}
