/* eslint-disable */
// @ts-nocheck
import { OnnxNodeBuilder } from './emitters.js';

export interface RNNOptions {
  returnSequences: boolean;
  returnState: boolean;
  goBackwards: boolean;
  stateful: boolean;
  unroll?: boolean;
  timeSteps?: number;
}

export interface LSTMOptions extends RNNOptions {
  recurrentActivation?: string;
  activation?: string;
}

export interface GRUOptions extends RNNOptions {
  resetAfter: boolean;
}

export interface BidirectionalOptions {
  mergeMode: 'concat' | 'sum' | 'mul' | 'ave' | null;
}

export function emitRNNBase(
  opType: 'RNN' | 'LSTM' | 'GRU',
  inputName: string,
  outputName: string,
  wName: string,
  rName: string,
  bName: string | undefined,
  initialStateNames: string[],
  name: string,
  options: RNNOptions & { linearBeforeReset?: number },
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];

  if (options.unroll && options.timeSteps) {
    // Statically unrolling the RNN cell into standard ONNX Math nodes
    // Example implementation for SimpleRNN: h_t = activation(W x_t + R h_{t-1} + B)
    let prevH = initialStateNames.length > 0 ? initialStateNames[0] : `${name}_initial_h`;

    if (initialStateNames.length === 0) {
      nodes.push({
        opType: 'Constant',
        inputs: [],
        outputs: [prevH],
        name: `${name}_init_h_const`,
        attributes: [{ name: 'value', f: 0.0, type: 'FLOAT' }], // pseudo zeroes
      });
    }

    const seqOutputs: string[] = [];

    for (let t = 0; t < options.timeSteps; t++) {
      const stepIndex = options.goBackwards ? options.timeSteps - 1 - t : t;
      const x_t = `${name}_x_${stepIndex}`;

      // 1. Extract x_t via Slice/Gather
      nodes.push({
        opType: 'Gather',
        inputs: [inputName, `${name}_idx_${t}`], // fake indices
        outputs: [x_t],
        name: `${name}_gather_${t}`,
        attributes: [{ name: 'axis', i: 1, type: 'INT' }], // axis 1 is time in layout=1
      });

      // 2. W x_t
      const wx_t = `${name}_wx_${t}`;
      nodes.push({
        opType: 'MatMul',
        inputs: [x_t, wName],
        outputs: [wx_t],
        name: `${name}_matmul_w_${t}`,
        attributes: [],
      });

      // 3. R h_{t-1}
      const rh_t = `${name}_rh_${t}`;
      nodes.push({
        opType: 'MatMul',
        inputs: [prevH, rName],
        outputs: [rh_t],
        name: `${name}_matmul_r_${t}`,
        attributes: [],
      });

      // 4. Add + Bias
      const add1_t = `${name}_add1_${t}`;
      nodes.push({
        opType: 'Add',
        inputs: [wx_t, rh_t],
        outputs: [add1_t],
        name: `${name}_add1_${t}`,
        attributes: [],
      });

      let h_t = add1_t;
      if (bName) {
        const add2_t = `${name}_add2_${t}`;
        nodes.push({
          opType: 'Add',
          inputs: [add1_t, bName],
          outputs: [add2_t],
          name: `${name}_add2_${t}`,
          attributes: [],
        });
        h_t = add2_t;
      }

      // 5. Activation
      const act_t = `${name}_h_${t}`;
      nodes.push({
        opType: 'Tanh', // default RNN activation
        inputs: [h_t],
        outputs: [act_t],
        name: `${name}_tanh_${t}`,
        attributes: [],
      });

      prevH = act_t;
      seqOutputs.push(act_t);
    }

    if (options.returnSequences) {
      nodes.push({
        opType: 'Concat',
        inputs: seqOutputs,
        outputs: [outputName],
        name: `${name}_concat_seq`,
        attributes: [{ name: 'axis', i: 1, type: 'INT' }],
      });
    } else {
      nodes.push({
        opType: 'Identity',
        inputs: [prevH],
        outputs: [outputName],
        name: `${name}_last_h`,
        attributes: [],
      });
    }

    return nodes;
  }

  // ONNX RNN inputs: X, W, R, B, sequence_lens, initial_h, initial_c (for LSTM), P (for peepholes)
  const inputs = [inputName, wName, rName];
  if (bName || initialStateNames.length > 0) {
    inputs.push(bName || ''); // B
  }
  inputs.push(''); // sequence_lens (optional)

  for (const stateName of initialStateNames) {
    inputs.push(stateName);
  }

  const direction = options.goBackwards ? 'reverse' : 'forward';

  // Outputs for ONNX RNNs: Y, Y_h, Y_c (for LSTM)
  // Y has shape [seq_length, num_directions, batch_size, hidden_size]
  const yOut = name + '_Y';
  const yhOut = name + '_Y_h';
  const ycOut = name + '_Y_c';

  const rnnOutputs = [yOut, yhOut];
  if (opType === 'LSTM') {
    rnnOutputs.push(ycOut);
  }

  const attributes = [
    { name: 'direction', s: direction, type: 'STRING' },
    // Keras RNNs expect input as [batch, seq, feature], ONNX expects [seq, batch, feature] by default
    // if opset 14+, layout=1 handles [batch, seq, feature]. Let's assume layout=1 is used.
    { name: 'layout', i: 1, type: 'INT' },
  ];

  if (options.linearBeforeReset !== undefined) {
    attributes.push({ name: 'linear_before_reset', i: options.linearBeforeReset, type: 'INT' });
  }

  nodes.push({
    opType,
    inputs,
    outputs: rnnOutputs,
    name,
    attributes,
  });

  // Now handle return_sequences and return_state
  // In Keras with layout=1, Y is [batch, seq, num_directions, hidden_size].
  // We need to squeeze the num_directions dimension (which is 1 for non-bidirectional).
  const squeezedY = name + '_Y_squeezed';
  nodes.push({
    opType: 'Squeeze',
    inputs: [yOut], // Assuming we squeeze axis 2 if rank is 4
    outputs: [squeezedY],
    name: name + '_squeeze_dir',
    attributes: [],
  });

  if (options.returnSequences) {
    // Just use the full sequence
    nodes.push({
      opType: 'Identity',
      inputs: [squeezedY],
      outputs: [outputName],
      name: name + '_seq_out',
      attributes: [],
    });
  } else {
    // Get the last state (or just use Y_h)
    // Y_h is [num_directions, batch_size, hidden_size]
    const squeezedYh = name + '_Yh_squeezed';
    nodes.push({
      opType: 'Squeeze',
      inputs: [yhOut], // squeeze axis 0
      outputs: [squeezedYh],
      name: name + '_squeeze_yh',
      attributes: [],
    });

    nodes.push({
      opType: 'Identity',
      inputs: [squeezedYh],
      outputs: [outputName],
      name: name + '_last_state_out',
      attributes: [],
    });
  }

  return nodes;
}

export function emitBidirectional(
  opType: 'RNN' | 'LSTM' | 'GRU',
  inputName: string,
  outputName: string,
  forwardWName: string,
  forwardRName: string,
  forwardBName: string | undefined,
  backwardWName: string,
  backwardRName: string,
  backwardBName: string | undefined,
  initialStateNames: string[], // [forward_states, backward_states]
  name: string,
  options: BidirectionalOptions & RNNOptions & { linearBeforeReset?: number },
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];

  const inputs = [inputName, forwardWName, forwardRName];
  if (forwardBName || initialStateNames.length > 0) {
    inputs.push(forwardBName || '');
  }
  inputs.push(''); // sequence_lens

  for (const stateName of initialStateNames) {
    inputs.push(stateName);
  }

  const yOut = name + '_Y';
  const yhOut = name + '_Y_h';
  const ycOut = name + '_Y_c';

  const rnnOutputs = [yOut, yhOut];
  if (opType === 'LSTM') rnnOutputs.push(ycOut);

  const attributes = [
    { name: 'direction', s: 'bidirectional', type: 'STRING' },
    { name: 'layout', i: 1, type: 'INT' },
  ];

  if (options.linearBeforeReset !== undefined) {
    attributes.push({ name: 'linear_before_reset', i: options.linearBeforeReset, type: 'INT' });
  }

  nodes.push({
    opType,
    inputs,
    outputs: rnnOutputs,
    name,
    attributes,
  });

  const mergeOut = name + '_merged';
  let seqOrStateOut = options.returnSequences ? yOut : yhOut;

  const splitForward = name + '_split_fwd';
  const splitBackward = name + '_split_bwd';

  if (options.mergeMode === 'concat') {
    nodes.push({
      opType: 'Reshape',
      inputs: [seqOrStateOut, name + '_reshape_target'],
      outputs: [outputName],
      name: name + '_merge_concat',
      attributes: [],
    });
  } else if (options.mergeMode === 'sum') {
    nodes.push({
      opType: 'ReduceSum',
      inputs: [seqOrStateOut],
      outputs: [outputName],
      name: name + '_merge_sum',
      attributes: [
        { name: 'axes', ints: [options.returnSequences ? 2 : 0], type: 'INTS' },
        { name: 'keepdims', i: 0, type: 'INT' },
      ],
    });
  } else if (options.mergeMode === 'ave') {
    nodes.push({
      opType: 'ReduceMean',
      inputs: [seqOrStateOut],
      outputs: [outputName],
      name: name + '_merge_ave',
      attributes: [
        { name: 'axes', ints: [options.returnSequences ? 2 : 0], type: 'INTS' },
        { name: 'keepdims', i: 0, type: 'INT' },
      ],
    });
  } else if (options.mergeMode === 'mul') {
    nodes.push({
      opType: 'Identity',
      inputs: [seqOrStateOut],
      outputs: [outputName],
      name: name + '_merge_mul_stub',
      attributes: [],
    });
  }

  return nodes;
}

export function reorderLSTMGates(weights: Float32Array, hiddenSize: number): Float32Array {
  const out = new Float32Array(weights.length);
  const numChunks = weights.length / (4 * hiddenSize);
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const offset = chunk * 4 * hiddenSize;
    out.set(weights.subarray(offset, offset + hiddenSize), offset); // i -> i
    out.set(
      weights.subarray(offset + 3 * hiddenSize, offset + 4 * hiddenSize),
      offset + hiddenSize,
    ); // o -> o
    out.set(
      weights.subarray(offset + hiddenSize, offset + 2 * hiddenSize),
      offset + 2 * hiddenSize,
    ); // f -> f
    out.set(
      weights.subarray(offset + 2 * hiddenSize, offset + 3 * hiddenSize),
      offset + 3 * hiddenSize,
    ); // c -> c
  }
  return out;
}

export function reorderGRUGates(weights: Float32Array, hiddenSize: number): Float32Array {
  return weights.slice();
}
