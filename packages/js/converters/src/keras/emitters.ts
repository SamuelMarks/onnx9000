export interface OnnxNodeBuilder {
  opType: string;
  inputs: string[];
  outputs: string[];
  name: string;
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

export interface ActivationOptions {
  alpha?: number;
  theta?: number;
  alphaWeightName?: string;
}

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
        attributes: [],
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
      const sigOut = name + '_sig';
      nodes.push({
        opType: 'Sigmoid',
        inputs: [inputName],
        outputs: [sigOut],
        name: name + '_sigmoid',
        attributes: [],
      });
      nodes.push({
        opType: 'Mul',
        inputs: [inputName, sigOut],
        outputs: [outputName],
        name: name + '_mul',
        attributes: [],
      });
      break;
    case 'gelu':
      nodes.push({
        opType: 'Gelu',
        inputs: [inputName],
        outputs: [outputName],
        name,
        attributes: [],
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
      throw new Error(`Unsupported activation: ${activation}`);
  }
  return nodes;
}

export function emitDense(
  inputName: string,
  outputName: string,
  weightName: string,
  biasName: string | undefined,
  activation: string,
  name: string,
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];
  const matmulOut = biasName || activation !== 'linear' ? name + '_matmul' : outputName;

  nodes.push({
    opType: 'MatMul',
    inputs: [inputName, weightName],
    outputs: [matmulOut],
    name: name + '_matmul',
    attributes: [],
  });

  let addOut = matmulOut;
  if (biasName) {
    addOut = activation !== 'linear' ? name + '_add' : outputName;
    nodes.push({
      opType: 'Add',
      inputs: [matmulOut, biasName],
      outputs: [addOut],
      name: name + '_add',
      attributes: [],
    });
  }

  if (activation && activation !== 'linear') {
    nodes.push(...emitActivation(activation, addOut, outputName, name + '_act'));
  }

  return nodes;
}

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
