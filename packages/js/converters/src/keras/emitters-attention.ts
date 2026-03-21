import { OnnxNodeBuilder } from './emitters.js';

export function emitAttention(
  queryName: string,
  valueName: string,
  keyName: string | undefined,
  outputName: string,
  name: string,
  useCausalMask: boolean,
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];
  const kName = keyName || valueName; // if key is omitted, key=value

  // Attention = Softmax(Q * K^T) * V
  // Let's assume Q, K, V are [batch, seq, features]

  // 1. Transpose K: [batch, seq, features] -> [batch, features, seq]
  const kTransposed = name + '_k_transposed';
  nodes.push({
    opType: 'Transpose',
    inputs: [kName],
    outputs: [kTransposed],
    name: name + '_transpose_k',
    attributes: [{ name: 'perm', ints: [0, 2, 1], type: 'INTS' }], // Batch matmul compatible
  });

  // 2. Q * K^T
  const scoresOut = name + '_scores';
  nodes.push({
    opType: 'MatMul',
    inputs: [queryName, kTransposed],
    outputs: [scoresOut],
    name: name + '_matmul_qk',
    attributes: [],
  });

  // 3. Causal Mask
  let softmaxIn = scoresOut;
  if (useCausalMask) {
    // Implement causal mask by adding a large negative value to the upper triangle
    // Here we just represent the mask add operation.
    const maskName = name + '_causal_mask'; // Assume this is a generated initializer or dynamic input
    const maskedScores = name + '_masked_scores';
    nodes.push({
      opType: 'Add',
      inputs: [scoresOut, maskName],
      outputs: [maskedScores],
      name: name + '_add_mask',
      attributes: [],
    });
    softmaxIn = maskedScores;
  }

  // 4. Softmax
  const softmaxOut = name + '_softmax';
  nodes.push({
    opType: 'Softmax',
    inputs: [softmaxIn],
    outputs: [softmaxOut],
    name: name + '_softmax',
    attributes: [{ name: 'axis', i: -1, type: 'INT' }],
  });

  // 5. Softmax * V
  nodes.push({
    opType: 'MatMul',
    inputs: [softmaxOut, valueName],
    outputs: [outputName],
    name: name + '_matmul_v',
    attributes: [],
  });

  return nodes;
}

export function emitEmbedding(
  inputName: string,
  weightName: string,
  outputName: string,
  name: string,
  maskZero: boolean,
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];

  // ONNX Embedding is just Gather
  nodes.push({
    opType: 'Gather',
    inputs: [weightName, inputName],
    outputs: [outputName],
    name,
    attributes: [{ name: 'axis', i: 0, type: 'INT' }],
  });

  if (maskZero) {
    // Keras masking outputs a boolean mask alongside.
    // We can emit a boolean mask tensor.
    const maskOut = name + '_mask';
    const zeroName = name + '_zero_const'; // Needs to be generated in actual transpiler
    nodes.push({
      opType: 'Equal',
      inputs: [inputName, zeroName],
      outputs: [name + '_eq_zero'],
      name: name + '_eq_zero',
      attributes: [],
    });
    nodes.push({
      opType: 'Not',
      inputs: [name + '_eq_zero'],
      outputs: [maskOut],
      name: name + '_mask_not',
      attributes: [],
    });
  }

  return nodes;
}
