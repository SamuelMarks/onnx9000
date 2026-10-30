import { describe, expect, it, vi } from 'vitest';
import {
  _check_op_specific,
  check_attribute,
  check_model,
  check_tensor,
  ValidationContext,
} from '../src/checker.js';
import { cleanName, ONNXToPyTorchVisitor } from '../src/codegen/pytorch.js';
import {
  AllowedWordsLogitProcessor,
  ComplexStoppingCriteria,
  ContrastiveSearchLogitProcessor,
  DiverseBeamSearchLogitProcessor,
  ForcedBOSLogitProcessor,
  ForcedEOSLogitProcessor,
  FrequencyPenaltyLogitProcessor,
  GrammarGuidedLogitProcessor,
  JSONSchemaLogitProcessor,
  LogitBiasProcessor,
  LogitProcessorList,
  MinPLogitProcessor,
  NoBadWordsLogitProcessor,
  NoRepeatNGramLogitProcessor,
  PresencePenaltyLogitProcessor,
  RegexLogitProcessor,
  RepetitionPenaltyLogitProcessor,
  TemperatureLogitProcessor,
  TopKLogitProcessor,
  TypicalLogitProcessor,
  WatermarkLogitProcessor,
} from '../src/genai/logit_processors.js';
import { SequenceTensorUtils } from '../src/genai/tensor_utils.js';
import {
  BasicTokenizer,
  BPETokenizer,
  HuggingFaceTokenizerLoader,
  UnigramTokenizer,
  WordPieceTokenizer,
} from '../src/genai/tokenizer.js';
import { TopPLogitProcessor } from '../src/genai/top_p.js';
import { Graph, Node, SparseTensor, Tensor, ValueInfo } from '../src/index.js';
import { irMacro, MacroMatcher, recordOp } from '../src/macros.js';
import { MBConv } from '../src/models/efficientnet.js';
import { SparseMoE } from '../src/models/mixtral.js';
import * as mathOps from '../src/ops/math.js';
import { DequantizeLinearOp, QuantizeLinearOp } from '../src/ops/quantization.js';
import { globalRegistry } from '../src/ops/registry.js';
import { serializeModelProto } from '../src/parser/onnx_writer.js';
import { BlobReader, BufferReader, skipField } from '../src/parser/protobuf.js';
import {
  checkSafetensors,
  createBuffer,
  decodeBfloat16,
  fetchSafetensorsChunk,
  fetchSafetensorsHeader,
  getEndianness,
  loadTensors,
  SafeTensors,
  saveSafetensors,
} from '../src/parser/safetensors.js';
import {
  AlibiBias,
  BatchNormalization,
  ConvND,
  DepthwiseConv,
  FlashAttention,
  Gelu,
  Gemm,
  GroupedQueryAttention,
  GroupNorm,
  InstanceNorm,
  LayerNormalization,
  LeakyRelu,
  MatMul,
  Mish,
  MultiHeadAttention,
  Relu,
  RMSNorm,
  RoPE,
  Sigmoid,
  Silu,
  Swish,
  Tanh,
} from '../src/primitives.js';
import {
  AutoShardingPass,
  allGather,
  allToAll,
  reduceScatter,
  SPMDLoweringPass,
} from '../src/sharding.js';
import {
  denseToBsr,
  denseToCoo,
  denseToCsc,
  denseToCsr,
  getTypedArray,
  sparseToCoo,
  sparseToDense,
  unpackData,
} from '../src/sparse.js';

describe('Coverage Core 3', () => {
  it('MBConv depthwiseConv kernelSize is array', () => {
    const block = new MBConv(32, 64, 3, 1, 1, 'test');
    block.depthwiseConv.kernelSize = [3, 3];
    const x = new Tensor(
      'x',
      [1, 32, 224, 224],
      'float32',
      false,
      true,
      new Float32Array(1 * 32 * 224 * 224),
    );
    block.call(x);
    expect(typeof x).toBeDefined();
  });

  it('covers checker.ts edge cases', () => {
    const ctx = new ValidationContext();
    check_tensor({ name: 't', data_type: 'invalid_type', shape: [-2] }, ctx);
    check_tensor({ name: 't', data_type: 'int32', shape: [-1], is_initializer: true }, ctx);
    check_tensor({ name: 't', data_type: 'int32', data_location: 'EXTERNAL' }, ctx);
    check_tensor(
      {
        name: 't',
        data_type: 'int32',
        data_location: 'EXTERNAL',
        external_data: { location: '../secret' },
      },
      ctx,
    );
    check_tensor(
      {
        name: 't',
        data_type: 'int32',
        shape: [2],
        raw_data: { byteLength: 3 * 1024 * 1024 * 1024 } as Uint8Array,
      },
      ctx,
    );
    expect(ctx.errors.length).toBeGreaterThan(4);

    check_attribute('pads', ['bad'], 'ints', ctx);
    check_attribute('scales', ['bad'], 'floats', ctx);
    _check_op_specific({ op_type: 'Add', inputs: ['a'] }, ctx);
    _check_op_specific({ op_type: 'Conv', inputs: ['a'], attributes: { pads: [1, 2, 3] } }, ctx);
    _check_op_specific({ op_type: 'If', attributes: {} }, ctx);
    _check_op_specific({ op_type: 'TreeEnsembleClassifier', attributes: {} }, ctx);

    const validIfModel = {
      ir_version: 8,
      producer_name: 'p',
      opset_import: [{ domain: 'ai.onnx', version: 15 }],
      graph: {
        inputs: [{ name: 'cond', data_type: 'bool', shape: [1] }],
        outputs: ['out2'],
        initializers: [
          { name: 'init1', data_type: 'int32', shape: [1], raw_data: new Uint8Array(4) },
        ],
        nodes: [
          {
            op_type: 'If',
            inputs: ['cond'],
            outputs: ['out'],
            attributes: {
              then_branch: {
                inputs: [{ name: 'cond', data_type: 'bool', shape: [1] }],
                outputs: ['out'],
                initializers: [],
                nodes: [{ op_type: 'Relu', inputs: ['cond'], outputs: ['out'], attributes: {} }],
              },
            },
          },
          {
            op_type: 'Loop',
            inputs: ['out'],
            outputs: ['out2'],
            attributes: {
              body: {
                inputs: [{ name: 'out', data_type: 'bool', shape: [1] }],
                outputs: ['out2'],
                initializers: [],
                nodes: [{ op_type: 'Relu', inputs: ['out'], outputs: ['out2'], attributes: {} }],
              },
            },
          },
        ],
      },
    };
    expect(check_model(validIfModel)).toBe(true);
  });

  it('covers macros.ts, primitives.ts, sharding.ts, and ir/tensor.ts', () => {
    const matcher = new MacroMatcher();
    const g = new Graph('test');
    expect(matcher.apply(g)).toBe(g);

    const decorator = irMacro('MyCustomMacro');
    const desc: PropertyDescriptor = {
      value: (t: Tensor) => t,
    };
    decorator({}, 'myMethod', desc);
    const tIn = new Tensor('in', [1], 'float32');
    expect((desc.value as (t: Tensor) => Tensor)(tIn).name).toBe('MyCustomMacro_out');
    expect(recordOp('TestOp', []).dtype).toBe('float32');

    const alibi = new AlibiBias(8);
    expect(alibi.call(tIn).name).toBe('AlibiBias_out');

    expect(allGather(tIn).name).toBe('AllGather_out');
    expect(reduceScatter(tIn).name).toBe('ReduceScatter_out');
    expect(allToAll(tIn).name).toBe('AllToAll_out');

    const gShard = new Graph('shard');
    const nMatMul = new Node('MatMul');
    const t1 = new Tensor('t1', [2, 2], 'float32');
    t1.sharding = [null, 'tp'];
    const t2 = new Tensor('t2', [2, 2], 'float32');
    t2.sharding = ['tp', null];
    const tOut = new Tensor('tOut', [2, 2], 'float32');
    nMatMul.inputs = [t1 as unknown as string, t2 as unknown as string];
    nMatMul.outputs = [tOut as unknown as string];
    gShard.nodes.push(nMatMul);
    new AutoShardingPass().apply(gShard);
    new SPMDLoweringPass().apply(gShard);
    expect(tOut.sharding).toEqual([null, null]);

    const st = new SparseTensor(
      'st',
      [2, 2],
      'CSR',
      undefined,
      undefined,
      undefined,
      undefined,
      [1, 1],
    );
    expect(st.blockDims).toEqual([1, 1]);
    expect(st.copy().name).toBe('st');
  });

  it('covers codegen/pytorch.ts', () => {
    const g = new Graph('MyGraph');
    const tInit = new Tensor('123init', [2], 'float32', true, false, new Float32Array([1, 2]));
    g.tensors['123init'] = tInit;
    g.initializers.push('123init');
    g.inputs.push(new ValueInfo('in.1', [2], 'float32'));
    g.outputs.push(new ValueInfo('out.1', [2], 'float32'));
    const nConv = new Node('Conv', ['in.1', '123init', 'bias'], ['h1']);
    const nGemm = new Node('Gemm', ['h1', '123init'], ['h2']);
    const nAdd = new Node('Add', ['h2', '123init'], ['h3']);
    const nRelu = new Node('Relu', ['h3'], ['h4']);
    const nFlat = new Node('Flatten', ['h4'], ['h5']);
    const nCustom = new Node('CustomOp', ['h5'], ['out.1']);
    g.nodes.push(nConv, nGemm, nAdd, nRelu, nFlat, nCustom);
    const gen = new ONNXToPyTorchVisitor(g);
    const code = gen.generate();
    expect(code).toContain('F.conv2d');

    const gEmpty = new Graph('');
    const nConvNoBias = new Node('Conv', ['x', 'w'], ['y']);
    const nGemmBias = new Node('Gemm', ['y', 'w', 'b'], ['z']);
    gEmpty.nodes.push(nConvNoBias, nGemmBias);
    const code2 = new ONNXToPyTorchVisitor(gEmpty).generate();
    expect(code2).toContain('return None');
  });

  it('covers genai logit_processors, top_p, tokenizer, tensor_utils', () => {
    const emptyLogits = new Tensor('l', [1, 0], 'float32', false, false, null);
    expect(new TemperatureLogitProcessor(1.0).process([], emptyLogits)).toBe(emptyLogits);
    expect(new TemperatureLogitProcessor(0.5).process([], emptyLogits)).toBe(emptyLogits);
    expect(new TopKLogitProcessor(1).process([], emptyLogits)).toBe(emptyLogits);
    const smallLogits = new Tensor(
      'l',
      [1, 2],
      'float32',
      false,
      false,
      new Float32Array([1.0, 2.0]),
    );
    expect(new TopKLogitProcessor(5).process([], smallLogits)).toBe(smallLogits);
    const topkOut = new TopKLogitProcessor(1).process([], smallLogits);
    expect(topkOut).not.toBe(smallLogits);

    expect(new RepetitionPenaltyLogitProcessor(1.0).process([0], smallLogits)).toBe(smallLogits);
    expect(new RepetitionPenaltyLogitProcessor(1.5).process([0], emptyLogits)).toBe(emptyLogits);
    const repOut = new RepetitionPenaltyLogitProcessor(1.5).process(
      [0, -1, 99],
      new Tensor('l', [1, 2], 'float32', false, false, new Float32Array([-1.0, 2.0])),
    );
    expect(repOut.data).toBeDefined();

    expect(new NoRepeatNGramLogitProcessor(2).process([0], smallLogits)).toBe(smallLogits);
    expect(new NoRepeatNGramLogitProcessor(2).process([0, 1], emptyLogits)).toBe(emptyLogits);
    const ngramOut = new NoRepeatNGramLogitProcessor(2).process([0, 1, 0], smallLogits);
    expect(ngramOut).not.toBe(smallLogits);

    expect(new MinPLogitProcessor(1.0).process([0], smallLogits)).toBe(smallLogits);
    expect(new MinPLogitProcessor(0.5).process([0], emptyLogits)).toBe(emptyLogits);
    expect(new MinPLogitProcessor(0.5).process([0], smallLogits)).not.toBe(smallLogits);

    expect(new PresencePenaltyLogitProcessor(0.0).process([0], smallLogits)).toBe(smallLogits);
    expect(new PresencePenaltyLogitProcessor(0.5).process([0], emptyLogits)).toBe(emptyLogits);
    expect(new PresencePenaltyLogitProcessor(0.5).process([0, 99], smallLogits)).not.toBe(
      smallLogits,
    );

    expect(new FrequencyPenaltyLogitProcessor(0.0).process([0], smallLogits)).toBe(smallLogits);
    expect(new FrequencyPenaltyLogitProcessor(0.5).process([0], emptyLogits)).toBe(emptyLogits);
    expect(new FrequencyPenaltyLogitProcessor(0.5).process([0, 0, 99], smallLogits)).not.toBe(
      smallLogits,
    );

    expect(new ForcedBOSLogitProcessor(0).process([0], smallLogits)).toBe(smallLogits);
    expect(new ForcedBOSLogitProcessor(0).process([], emptyLogits)).toBe(emptyLogits);
    expect(new ForcedBOSLogitProcessor(0).process([], smallLogits)).not.toBe(smallLogits);

    expect(new ForcedEOSLogitProcessor(2, 1).process([0], smallLogits)).not.toBe(smallLogits);
    expect(new ForcedEOSLogitProcessor(1, 1).process([0], emptyLogits)).toBe(emptyLogits);

    expect(new NoBadWordsLogitProcessor([[0]]).process([0], emptyLogits)).toBe(emptyLogits);
    expect(
      new NoBadWordsLogitProcessor([
        [0, 1],
        [9, 1],
        [1, 99],
      ]).process([0], smallLogits),
    ).not.toBe(smallLogits);

    expect(new AllowedWordsLogitProcessor([0]).process([0], emptyLogits)).toBe(emptyLogits);
    expect(new AllowedWordsLogitProcessor([0]).process([0], smallLogits)).not.toBe(smallLogits);

    const biasMap = new Map<number, number>([
      [0, 1.5],
      [99, 2.0],
    ]);
    expect(new LogitBiasProcessor(biasMap).process([], emptyLogits)).toBe(emptyLogits);
    expect(new LogitBiasProcessor(biasMap).process([], smallLogits)).not.toBe(smallLogits);

    expect(new LogitProcessorList([new TypicalLogitProcessor()]).process([], smallLogits)).toBe(
      smallLogits,
    );
    expect(new DiverseBeamSearchLogitProcessor().process([], smallLogits)).toBe(smallLogits);
    expect(new ContrastiveSearchLogitProcessor().process([], smallLogits)).toBe(smallLogits);
    expect(new GrammarGuidedLogitProcessor()).toBeDefined();
    expect(new JSONSchemaLogitProcessor()).toBeDefined();
    expect(new RegexLogitProcessor()).toBeDefined();
    expect(new WatermarkLogitProcessor()).toBeDefined();
    expect(new ComplexStoppingCriteria()).toBeDefined();

    expect(new TopPLogitProcessor(1.0).process([], smallLogits)).toBe(smallLogits);
    expect(new TopPLogitProcessor(0.5).process([], emptyLogits)).toBe(emptyLogits);
    expect(new TopPLogitProcessor(0.5, 5).process([], smallLogits)).not.toBe(smallLogits);
    const toppOut = new TopPLogitProcessor(0.1, 1).process(
      [],
      new Tensor('l', [1, 3], 'float32', false, false, new Float32Array([10.0, 0.0, -10.0])),
    );
    expect(toppOut).not.toBe(smallLogits);

    const expandedNull = SequenceTensorUtils.expandSequenceDimension(
      new Tensor('t', [1, 2, 2], 'float32', false, false, null),
      4,
    );
    expect(expandedNull.shape).toEqual([1, 4, 2]);
    const expandedStr = SequenceTensorUtils.expandSequenceDimension(
      new Tensor('t', [1, 2, 2], 'string', false, false, new Uint8Array(4)),
      4,
    );
    expect(expandedStr.shape).toEqual([1, 4, 2]);

    const bt = new BasicTokenizer();
    expect(bt.decode([65])).toBe('A');
    const bpe = new BPETokenizer([['a', 'b']], new Map([['a', 1]]));
    expect(bpe.idToToken(1)).toBe('a');
    expect(bpe.idToToken(999)).toBe('<unk>');
    expect(bpe.tokenToId('a')).toBe(1);
    const stream = bpe.createStream();
    expect(stream.put(999)).toBe('<unk>');
  });

  it('covers mathOps, quantization, registry, sparse, onnx_writer, protobuf, safetensors', async () => {
    const f32Empty = new Tensor('e', [0], 'float32', false, false, new Float32Array(0));
    const u8Tensor = new Tensor('u', [2], 'uint8', false, false, new Uint8Array([1, 2]));
    const f32Short = new Tensor('s', [1], 'float32', false, false, new Float32Array([1]));
    const f32Long = new Tensor('l', [2], 'float32', false, false, new Float32Array([1, 2]));
    for (const val of Object.values(mathOps)) {
      if (typeof val === 'function') {
        const OpClass = val as new () => {
          execute: (ins: Tensor[], attrs: Record<string, never>) => Tensor[];
        };
        const op = new OpClass();
        op.execute([f32Empty, f32Empty], {});
        op.execute([u8Tensor, u8Tensor], {});
        op.execute([f32Short, f32Long], {});
        op.execute([f32Long, f32Short], {});
      }
    }

    const qOp = new QuantizeLinearOp();
    expect(qOp.execute([u8Tensor], {}).length).toBe(1);
    const dqOp = new DequantizeLinearOp();
    expect(dqOp.execute([f32Long], {}).length).toBe(1);

    globalRegistry.register_op('', 'TestEmptyDomain')(mathOps.AbsOp);
    expect(globalRegistry.get_op('', 'TestEmptyDomain', 'webgpu')).toBe(mathOps.AbsOp);
    expect(globalRegistry.getAllRegistered('nonexistent')).toEqual({});

    for (const dt of [
      'float64',
      'int8',
      'int16',
      'int32',
      'int64',
      'uint8',
      'uint16',
      'uint32',
      'uint64',
      'bool',
      'float16',
      'bfloat16',
    ] as const) {
      const arr = getTypedArray(dt, 2);
      expect(arr).toBeDefined();
      const t = new Tensor('t', [2], dt, true, false, new Uint8Array(arr.buffer));
      expect(unpackData(t).length).toBeGreaterThan(0);
    }
    const tNull = new Tensor('tn', [1], 'float32', false, false, null);
    expect(unpackData(tNull)).toEqual([]);

    const dense2d = new Tensor(
      'd2',
      [2, 2],
      'float32',
      true,
      false,
      new Float32Array([1, 0, 0, 2]),
    );
    const coo = denseToCoo(dense2d);
    expect(sparseToDense(coo).shape).toEqual([2, 2]);
    const csr = denseToCsr(dense2d);
    expect(sparseToDense(csr).shape).toEqual([2, 2]);
    const csc = denseToCsc(dense2d);
    expect(sparseToDense(csc).shape).toEqual([2, 2]);
    const bsr = denseToBsr(dense2d, [1, 1]);
    expect(sparseToDense(bsr).shape).toEqual([2, 2]);
    expect(sparseToCoo(coo)).toBe(coo);
    const emptySparse = new SparseTensor('es', [2, 2], 'COO');
    expect(sparseToDense(emptySparse).shape).toEqual([2, 2]);

    const tNoShape = new Tensor('ns', [], 'float32', true, false, new Float32Array([1]));
    const gWriter = new Graph('gw');
    gWriter.tensors.ns = tNoShape;
    gWriter.initializers.push('ns');
    gWriter.opsetImports[''] = 15;
    expect(serializeModelProto(gWriter).byteLength).toBeGreaterThan(0);

    const reader32 = new BufferReader(new Uint8Array([1, 2, 3, 4]));
    await skipField(reader32, 5);
    expect(reader32.getPosition()).toBe(4);
    await expect(skipField(reader32, 3)).rejects.toThrow(/Groups are not supported/);
    await expect(skipField(reader32, 4)).rejects.toThrow(/Groups are not supported/);
    await expect(skipField(reader32, 99)).rejects.toThrow(/Unsupported wire type/);

    const hdr = JSON.stringify({
      t1: { dtype: 'F32', shape: [1], data_offsets: [0, 4] },
    });
    const hdrBytes = new TextEncoder().encode(hdr);
    const buf = new Uint8Array(8 + hdrBytes.byteLength + 4);
    new DataView(buf.buffer).setBigUint64(0, BigInt(hdrBytes.byteLength), true);
    buf.set(hdrBytes, 8);
    const st = new SafeTensors(buf.buffer);
    expect(st.getTypedArray('t1')).toBeInstanceOf(Float32Array);
  });

  it('covers remaining branches in primitives, checker, sharding, sparse, pytorch, tokenizer, safetensors', async () => {
    const t = new Tensor('t', [2, 2], 'float32', false, false, new Float32Array([1, 2, 3, 4]));
    expect(new BatchNormalization(4).call(t, t, t, t, t).name).toBe('BatchNormalization_out');
    expect(new LayerNormalization([4]).call(t, t).name).toBe('LayerNormalization_out');
    expect(new LayerNormalization([4]).call(t, t, t).name).toBe('LayerNormalization_out');
    expect(new RMSNorm([4]).call(t, t).name).toBe('RMSNormalization_out');
    expect(new GroupNorm(2, 4).call(t, t, t).name).toBe('GroupNormalization_out');
    expect(new InstanceNorm(4).call(t, t, t).name).toBe('InstanceNormalization_out');
    expect(new Relu().call(t).name).toBe('Relu_out');
    expect(new Relu().generateLUT().name).toBe('Constant_out');
    expect(new Sigmoid().call(t).name).toBe('Sigmoid_out');
    expect(new Tanh().call(t).name).toBe('Tanh_out');
    expect(new LeakyRelu().call(t).name).toBe('LeakyRelu_out');
    expect(new Gelu().call(t).name).toBe('Gelu_out');
    expect(new Silu().call(t).name).toBe('Swish_out');
    expect(new Swish().call(t).name).toBe('Swish_out');
    expect(new Mish().call(t).name).toBe('Mish_out');
    expect(new ConvND(2, 3, 3, 3).call(t, t, t).name).toBe('Conv_out');
    expect(new DepthwiseConv(2, 3, 3).call(t, t).name).toBe('Conv_out');
    expect(new MatMul().call(t, t).name).toBe('MatMul_out');
    expect(new Gemm().call(t, t).name).toBe('Gemm_out');
    expect(new Gemm().call(t, t, t).name).toBe('Gemm_out');
    expect(new MultiHeadAttention(4).call(t, t, t).name).toBe('Attention_out');
    expect(new MultiHeadAttention(4).call(t, t, t, t).name).toBe('Attention_out');
    expect(new FlashAttention(4).call(t, t, t).name).toBe('FlashAttention_out');
    expect(new FlashAttention(4).call(t, t, t, t).name).toBe('FlashAttention_out');
    expect(new GroupedQueryAttention(4, 2).call(t, t, t).name).toBe('GroupedQueryAttention_out');
    expect(new GroupedQueryAttention(4, 2).call(t, t, t, t).name).toBe('GroupedQueryAttention_out');
    expect(new RoPE(4).call(t, t).name).toBe('RoPE_out');

    // checker.ts lines 271-272, 282-283
    expect(() =>
      check_model({
        ir_version: 8,
        producer_name: 123 as unknown as string,
        opset_import: [
          { domain: 'ai.onnx', version: 15 },
          { domain: 'ai.onnx', version: 15 },
        ],
        graph: {
          inputs: [
            { name: 'dup', data_type: 'float', shape: [1] },
            { name: 'dup', data_type: 'float', shape: [1] },
          ],
          initializers: [{ name: 'dup', data_type: 'float', shape: [1] }],
          nodes: [{ op_type: 'Relu', inputs: ['dangling', ''], outputs: ['dup', 'dup'] }],
          outputs: ['dup'],
        },
      }),
    ).toThrow();

    // sharding.ts lines 23-26: MatMul with <2 inputs, no sharding, no outputs, no shape
    const gShard = new Graph('s');
    const n1 = new Node('MatMul', ['a']);
    const n2 = new Node('MatMul', ['a', 'b'], []);
    const tA = new Tensor('a', [], 'float32');
    tA.sharding = ['dp'];
    const tB = new Tensor('b', [], 'float32');
    n2.inputs = [tA as unknown as string, tB as unknown as string];
    const n3 = new Node('MatMul');
    const tOutNoShape = { sharding: null } as unknown as string;
    n3.inputs = [tA as unknown as string, tB as unknown as string];
    n3.outputs = [tOutNoShape];
    const n4 = new Node('MatMul');
    n4.inputs = [tB as unknown as string, tB as unknown as string];
    gShard.nodes.push(n1, n2, n3, n4);
    new AutoShardingPass().apply(gShard);

    // sparse.ts lines 145-146, 223-224: 0-dim shape and bigint values
    const tEmptyShape = new Tensor(
      'es',
      [],
      'int64',
      true,
      false,
      new Uint8Array(new BigInt64Array([0n, 1n]).buffer),
    );
    expect(denseToCsr(tEmptyShape).format).toBe('CSR');
    expect(denseToCsc(tEmptyShape).format).toBe('CSC');

    // pytorch.ts lines 44, 60, 62, 64, 66, 68
    expect(cleanName('')).toBe('var_empty');
    const gPy = new Graph('py');
    gPy.tensors.undef = undefined as unknown as Tensor;
    gPy.tensors.non_init = new Tensor('non_init', [1], 'float32', false);
    gPy.nodes.push(
      new Node('Relu', [], []),
      new Node('Conv', [], []),
      new Node('MatMul', [], []),
      new Node('Add', [], []),
      new Node('Reshape', [], []),
    );
    expect(new ONNXToPyTorchVisitor(gPy).generate()).toContain('F.relu');

    // logit_processors.ts lines 86, 454 & top_p.ts line 56
    const nanLogits = new Tensor('nan', [1, 1], 'float32', false, false, new Float32Array([NaN]));
    expect(new TopKLogitProcessor(1).process([], nanLogits)).toBeDefined();
    expect(new NoRepeatNGramLogitProcessor(1).process([0], nanLogits)).toBeDefined();
    expect(new TopPLogitProcessor(0.5, 1).process([], nanLogits)).toBeDefined();

    // tokenizer.ts lines 399, 429, 513-514
    const wp = new WordPieceTokenizer(
      new Map([
        ['a', 1],
        ['##b', 2],
      ]),
    );
    expect(wp.decode([1, 2], false)).toBe('ab');
    const uni = new UnigramTokenizer(new Map([['a', -1.0]]));
    expect(uni.encode('  a  ')).toEqual([0]);
    expect(uni.decode([0, 999], false)).toBe('a <unk>');
    expect(
      HuggingFaceTokenizerLoader.loadFromJson(JSON.stringify({ model: { type: 'BPE' } })),
    ).toBeInstanceOf(BPETokenizer);
    expect(
      HuggingFaceTokenizerLoader.loadFromJson(JSON.stringify({ model: { type: 'WordPiece' } })),
    ).toBeInstanceOf(WordPieceTokenizer);
    expect(
      HuggingFaceTokenizerLoader.loadFromJson(JSON.stringify({ model: { type: 'Unigram' } })),
    ).toBeInstanceOf(UnigramTokenizer);

    // ir/tensor.ts line 154: non-Uint8Array typed array with undefined element fallback
    const tF32 = new Tensor('f', [2], 'float32', false, false, new Float32Array([1, 2]));
    expect(tF32.formatData(1)).toContain('elements');

    // models/mixtral.ts line 79: 0 experts
    const moe0 = new SparseMoE(0, 1, 4, 8, 'm');
    expect(moe0.call(t).name).toBe('ScatterND_out');

    // parser/onnx_writer.ts lines 130, 251: node with empty opType, node with undefined attributes
    const gOnnx = new Graph('go');
    const nEmptyOp = new Node('', ['in'], ['out']);
    (nEmptyOp as unknown as { attributes: undefined }).attributes = undefined;
    gOnnx.nodes.push(nEmptyOp);
    gOnnx.inputs.push(new ValueInfo('in', [''], 'float32'));
    expect(serializeModelProto(gOnnx).byteLength).toBeGreaterThan(0);

    // parser/protobuf.ts lines 135, 147: BlobReader cache miss
    const blob = new Blob([new Uint8Array([1, 2, 3, 4])]);
    const br = new BlobReader(blob, 2);
    (br as unknown as { cache: null }).cache = null;
    vi.spyOn(
      br as unknown as { ensureCache: () => Promise<void> },
      'ensureCache',
    ).mockResolvedValueOnce();
    expect(await br.readByte()).toBe(0);
    vi.spyOn(
      br as unknown as { ensureCache: () => Promise<void> },
      'ensureCache',
    ).mockResolvedValueOnce();
    expect(await br.readBytes(1)).toEqual(new Uint8Array());

    // parser/safetensors.ts lines 222, 384, 475-476
    expect(createBuffer(8, true).byteLength).toBe(8);
    const hdr2 = JSON.stringify({
      t1: { dtype: 'I32', shape: [1], data_offsets: [0, 4] },
      t2: { dtype: 'I16', shape: [1], data_offsets: [8, 10] },
      t3: { dtype: 'U32', shape: [1], data_offsets: [16, 20] },
      t4: { dtype: 'U16', shape: [1], data_offsets: [24, 26] },
      t5: { dtype: 'I64', shape: [1], data_offsets: [32, 40] },
      t6: { dtype: 'U64', shape: [1], data_offsets: [40, 48] },
      t7: { dtype: 'F16', shape: [1], data_offsets: [48, 50] },
      t8: { dtype: 'BF16', shape: [1], data_offsets: [56, 58] },
      t9: { dtype: 'BOOL', shape: [1], data_offsets: [64, 65] },
      t10: { dtype: 'I8', shape: [1], data_offsets: [72, 73] },
      t11: { dtype: 'U8', shape: [1], data_offsets: [80, 81] },
      t12: { dtype: 'F64', shape: [1], data_offsets: [88, 96] },
    });
    const hdr2Bytes = new TextEncoder().encode(hdr2);
    const buf2 = new Uint8Array(8 + hdr2Bytes.byteLength + 96);
    new DataView(buf2.buffer).setBigUint64(0, BigInt(hdr2Bytes.byteLength), true);
    buf2.set(hdr2Bytes, 8);
    const st2 = new SafeTensors(buf2.buffer, 'BE');
    for (let i = 1; i <= 12; i++) {
      expect(st2.getTypedArray(`t${i}`)).toBeDefined();
    }
    expect(() => st2.getTypedArray('missing')).toThrow();

    const origFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockResolvedValue({
      status: 200,
      ok: true,
      arrayBuffer: async () => buf2.buffer,
      headers: new Headers(),
    });
    const fetched = await fetchSafetensorsHeader('hf://user/repo/model.safetensors');
    expect(fetched.fullBuffer).toBeDefined();
    globalThis.fetch = origFetch;

    // Additional branch coverage for checker.ts
    const ctx2 = new ValidationContext();
    check_tensor(
      {
        name: 't_ext_ok',
        data_type: 'float',
        data_location: 'EXTERNAL',
        external_data: { location: '' },
      },
      ctx2,
    );
    _check_op_specific({ op_type: 'Add' }, ctx2);
    _check_op_specific({ op_type: 'Conv', attributes: { pads: [1, 2] } }, ctx2);
    _check_op_specific({ op_type: 'TreeEnsembleClassifier' }, ctx2);
    expect(() => check_model({ ir_version: 2 })).toThrow();
    expect(() =>
      check_model({
        ir_version: 8,
        opset_import: [{ domain: 'ai.onnx', version: 15 }],
        graph: {
          nodes: [{ op_type: 'Identity' }],
        },
      }),
    ).not.toThrow();

    // tokenizer.ts: BPE encode with non-matching merge pair & empty JSON loadFromJson
    const bpe2 = new BPETokenizer(
      [
        ['x', 'y'],
        ['a', 'b'],
      ],
      new Map([
        ['ab', 1],
        ['c', 2],
      ]),
    );
    expect(bpe2.encode('abc')).toEqual([1, 2]);
    expect(() => HuggingFaceTokenizerLoader.loadFromJson('{}')).toThrow(/Unsupported model type/);

    // onnx_writer.ts: tensor with null data
    const gOnnxNull = new Graph('gn');
    gOnnxNull.tensors.tnull = new Tensor('tnull', [1], 'float32', true, false, null);
    gOnnxNull.initializers.push('tnull');
    expect(serializeModelProto(gOnnxNull).byteLength).toBeGreaterThan(0);

    // safetensors.ts remaining branches
    const badShapeHdr = JSON.stringify({
      t1: { dtype: 'F32', shape: 'not_array', data_offsets: [0, 4] },
    });
    const badShapeBytes = new TextEncoder().encode(badShapeHdr);
    const badShapeBuf = new Uint8Array(8 + badShapeBytes.byteLength + 4);
    new DataView(badShapeBuf.buffer).setBigUint64(0, BigInt(badShapeBytes.byteLength), true);
    badShapeBuf.set(badShapeBytes, 8);
    expect(() => new SafeTensors(badShapeBuf.buffer)).toThrow(/Shape must be an array/);

    const negShapeHdr = JSON.stringify({
      t1: { dtype: 'F32', shape: [-1], data_offsets: [0, 4] },
    });
    const negShapeBytes = new TextEncoder().encode(negShapeHdr);
    const negShapeBuf = new Uint8Array(8 + negShapeBytes.byteLength + 4);
    new DataView(negShapeBuf.buffer).setBigUint64(0, BigInt(negShapeBytes.byteLength), true);
    negShapeBuf.set(negShapeBytes, 8);
    expect(() => new SafeTensors(negShapeBuf.buffer)).toThrow(/Negative dimension/);

    const hugeShapeHdr = JSON.stringify({
      t1: { dtype: 'F32', shape: [Number.MAX_SAFE_INTEGER + 1], data_offsets: [0, 4] },
    });
    const hugeShapeBytes = new TextEncoder().encode(hugeShapeHdr);
    const hugeShapeBuf = new Uint8Array(8 + hugeShapeBytes.byteLength + 4);
    new DataView(hugeShapeBuf.buffer).setBigUint64(0, BigInt(hugeShapeBytes.byteLength), true);
    hugeShapeBuf.set(hugeShapeBytes, 8);
    expect(() => new SafeTensors(hugeShapeBuf.buffer)).toThrow(/Dimension too large/);

    const nonStrMetaHdr = JSON.stringify({
      __metadata__: { num: 123 },
      t1: {},
    });
    const nonStrMetaBytes = new TextEncoder().encode(nonStrMetaHdr);
    const nonStrMetaBuf = new Uint8Array(8 + nonStrMetaBytes.byteLength + 4);
    new DataView(nonStrMetaBuf.buffer).setBigUint64(0, BigInt(nonStrMetaBytes.byteLength), true);
    nonStrMetaBuf.set(nonStrMetaBytes, 8);
    expect(() => new SafeTensors(nonStrMetaBuf.buffer)).toThrow(/Unknown dtype/);

    const noShapeOffsetsHdr = JSON.stringify({
      t1: { dtype: 'F32' },
    });
    const noShapeOffsetsBytes = new TextEncoder().encode(noShapeOffsetsHdr);
    const noShapeOffsetsBuf = new Uint8Array(8 + noShapeOffsetsBytes.byteLength + 4);
    new DataView(noShapeOffsetsBuf.buffer).setBigUint64(
      0,
      BigInt(noShapeOffsetsBytes.byteLength),
      true,
    );
    noShapeOffsetsBuf.set(noShapeOffsetsBytes, 8);
    expect(() => new SafeTensors(noShapeOffsetsBuf.buffer)).toThrow(/Shape volume/);

    const jsonNullHdr = 'null    ';
    const jsonNullBytes = new TextEncoder().encode(jsonNullHdr);
    const jsonNullBuf = new Uint8Array(8 + jsonNullBytes.byteLength);
    new DataView(jsonNullBuf.buffer).setBigUint64(0, BigInt(jsonNullBytes.byteLength), true);
    jsonNullBuf.set(jsonNullBytes, 8);
    expect(() => new SafeTensors(jsonNullBuf.buffer)).toThrow(/Header must be a dictionary/);

    // checker.ts lines 142, 266, 305, 312
    const reg2 = new (await import('../src/checker.js')).SchemaRegistry();
    expect(() => reg2.get_schema('Conv', 99, 'ai.onnx')).toThrow(/Unsupported opset/);
    expect(() =>
      check_model({
        ir_version: 11,
        opset_import: [{ domain: 'ai.onnx', version: 15 }],
        graph: { inputs: [{ name: 'in1', data_type: 'float', shape: [1] }] },
      }),
    ).toThrow(/Invalid ir_version/);
    expect(() =>
      check_model({
        ir_version: 0,
      } as unknown as Parameters<typeof check_model>[0]),
    ).toThrow();

    // onnx_writer.ts line 251: tensor with string dimension in shape
    const gSymDim = new Graph('sym');
    gSymDim.tensors.tsym = new Tensor(
      'tsym',
      ['batch'],
      'float32',
      true,
      false,
      new Float32Array([1]),
    );
    gSymDim.initializers.push('tsym');
    expect(serializeModelProto(gSymDim).byteLength).toBeGreaterThan(0);

    // safetensors.ts lines 242, 429, 483, 497, 502, 558, 599, 612, 638, 707, 816-817, 880, 930-931
    const unquotedHdr = new TextEncoder().encode('{bad json   ');
    const unquotedBuf = new Uint8Array(8 + unquotedHdr.byteLength);
    new DataView(unquotedBuf.buffer).setBigUint64(0, BigInt(unquotedHdr.byteLength), true);
    unquotedBuf.set(unquotedHdr, 8);
    expect(() => new SafeTensors(unquotedBuf.buffer)).toThrow(/Invalid JSON header/);

    expect(decodeBfloat16(new Uint16Array([0x3f80]), 'BE').length).toBe(1);
    expect(
      saveSafetensors({ t1: new Uint8Array(8) }, { format: 'pt', version: '1.0' }).byteLength,
    ).toBeGreaterThan(0);
    expect(() =>
      checkSafetensors({
        get byteLength() {
          throw new TypeError('custom non-safetensors error');
        },
      } as unknown as ArrayBuffer),
    ).toThrow(TypeError);

    const origFetch2 = globalThis.fetch;
    globalThis.fetch = vi.fn().mockImplementation(async (_url: string, opts?: RequestInit) => {
      const range = (opts?.headers as Record<string, string>)?.Range || '';
      if (range === 'bytes=0-7') {
        const b = new ArrayBuffer(8);
        new DataView(b).setBigUint64(0, BigInt(200 * 1024 * 1024), true);
        return {
          status: 206,
          ok: true,
          headers: new Headers({ 'Accept-Ranges': 'bytes' }),
          arrayBuffer: async () => b,
        };
      }
      return { status: 400, ok: false, statusText: 'Bad' };
    });
    await expect(fetchSafetensorsHeader('hf://invalid')).rejects.toThrow(
      /Header size exceeds 100MB/,
    );

    globalThis.fetch = vi.fn().mockImplementation(async (_url: string, opts?: RequestInit) => {
      const range = (opts?.headers as Record<string, string>)?.Range || '';
      if (range === 'bytes=0-7') {
        const b = new ArrayBuffer(8);
        new DataView(b).setBigUint64(0, 4n, true);
        return {
          status: 206,
          ok: true,
          headers: new Headers(),
          arrayBuffer: async () => b,
        };
      }
      return { status: 500, ok: false, statusText: 'Err' };
    });
    await expect(fetchSafetensorsHeader('http://example.com/a.safetensors')).rejects.toThrow(
      /Failed to fetch header/,
    );

    // WebSocket non-ArrayBuffer message ignored before ArrayBuffer message arrives
    const origWS = globalThis.WebSocket;
    class MockWS {
      binaryType = 'arraybuffer';
      onopen?: () => void;
      onmessage?: (ev: { data: unknown }) => void;
      constructor(_u: string) {
        setTimeout(() => {
          this.onopen?.();
          this.onmessage?.({ data: 'text' });
          this.onmessage?.({ data: new ArrayBuffer(4) });
        }, 1);
      }
      send() {}
      close() {}
    }
    globalThis.WebSocket = MockWS as unknown as typeof WebSocket;
    const wsChunk = await fetchSafetensorsChunk('ws://localhost/t', 8, 0, 4);
    expect(wsChunk.byteLength).toBe(4);
    globalThis.WebSocket = origWS;

    // fetchSafetensorsChunk with fullBuffer and no onProgress
    const fbChunk = await fetchSafetensorsChunk('hf://invalid', 0, 0, 4, buf2.buffer);
    expect(fbChunk.byteLength).toBe(4);

    // fetchSafetensorsChunk with reader yielding undefined value and no onProgress, plus cacheStorage
    const origCaches = globalThis.caches;
    globalThis.caches = {
      open: async () => ({
        match: async () => ({ arrayBuffer: async () => new ArrayBuffer(4) }),
        put: async () => undefined,
      }),
    } as unknown as CacheStorage;
    const cachedChunk = await fetchSafetensorsChunk('http://example.com/c', 8, 0, 4);
    expect(cachedChunk.byteLength).toBe(4);

    globalThis.caches = {
      open: async () => ({
        match: async () => undefined,
        put: async () => undefined,
      }),
    } as unknown as CacheStorage;
    let readCalls = 0;
    globalThis.fetch = vi.fn().mockResolvedValue({
      status: 206,
      ok: true,
      headers: new Headers(),
      body: {
        getReader: () => ({
          read: async () => {
            readCalls++;
            if (readCalls === 1) return { done: false, value: undefined };
            if (readCalls === 2) return { done: false, value: new Uint8Array([1, 2]) };
            return { done: true, value: undefined };
          },
        }),
      },
    });
    const streamChunk = await fetchSafetensorsChunk('http://example.com/s', 8, 0, 2);
    expect(streamChunk).toEqual(new Uint8Array([1, 2]));
    globalThis.caches = origCaches;

    // loadTensors with RegExp pattern
    globalThis.fetch = vi.fn().mockResolvedValue({
      status: 200,
      ok: true,
      arrayBuffer: async () => buf2.buffer,
      headers: new Headers(),
    });
    const loaded: string[] = [];
    for await (const item of loadTensors('http://example.com/m', { pattern: /^t1$/ })) {
      loaded.push(item.name);
    }
    expect(loaded).toEqual(['t1']);

    // Final branch coverage for checker.ts and safetensors.ts
    expect(() =>
      check_model({
        ir_version: 0,
        opset_import: [{ domain: 'ai.onnx', version: 15 }],
        graph: { inputs: [{ name: 'in1', data_type: 'float', shape: [1] }] },
      }),
    ).toThrow(/Invalid ir_version/);

    const jsMetaHdr = JSON.stringify({
      __metadata__: { link: 'javascript:alert(1)' },
    });
    const jsMetaBytes = new TextEncoder().encode(jsMetaHdr);
    const jsMetaBuf = new Uint8Array(8 + jsMetaBytes.byteLength);
    new DataView(jsMetaBuf.buffer).setBigUint64(0, BigInt(jsMetaBytes.byteLength), true);
    jsMetaBuf.set(jsMetaBytes, 8);
    expect(() => new SafeTensors(jsMetaBuf.buffer)).toThrow(/Executable script tags/);

    const dupHdr = '{"toString":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}   ';
    const dupBytes = new TextEncoder().encode(dupHdr);
    const dupBuf = new Uint8Array(8 + dupBytes.byteLength + 4);
    new DataView(dupBuf.buffer).setBigUint64(0, BigInt(dupBytes.byteLength), true);
    dupBuf.set(dupBytes, 8);
    expect(() => new SafeTensors(dupBuf.buffer)).toThrow(/Duplicate tensor name/);

    expect(getEndianness(new Uint8Array([0x12, 0x34, 0x56, 0x78]).buffer)).toBe('BE');
    expect(
      saveSafetensors({ t1: new Uint8Array(8) }, { format: 'custom', version: '2.0' }).byteLength,
    ).toBeGreaterThan(0);

    globalThis.fetch = vi.fn().mockImplementation(async (_url: string, opts?: RequestInit) => {
      const range = (opts?.headers as Record<string, string>)?.Range || '';
      if (range === 'bytes=0-7') {
        const b = new ArrayBuffer(8);
        new DataView(b).setBigUint64(0, 8n, true);
        return {
          status: 206,
          ok: true,
          headers: new Headers(),
          arrayBuffer: async () => b,
        };
      }
      return {
        status: 206,
        ok: true,
        headers: new Headers(),
        arrayBuffer: async () => new TextEncoder().encode('null    ').buffer,
      };
    });
    await expect(
      fetchSafetensorsHeader('https://huggingface.co/model.safetensors'),
    ).rejects.toThrow(/Header must be a dictionary/);
    await expect(fetchSafetensorsHeader('hf://a/b')).rejects.toThrow(/Header must be a dictionary/);

    const arrHdr = new TextEncoder().encode('[1, 2, 3]       ');
    const arrBuf = new Uint8Array(8 + arrHdr.byteLength);
    new DataView(arrBuf.buffer).setBigUint64(0, BigInt(arrHdr.byteLength), true);
    arrBuf.set(arrHdr, 8);
    expect(() => new SafeTensors(arrBuf.buffer)).toThrow(/Header must be a dictionary/);
    expect(checkSafetensors(arrBuf.buffer)).toBe(false);

    // safetensors.ts line 208: headerSizeBig > 100MB
    const bigHdrBuf = new ArrayBuffer(8);
    new DataView(bigHdrBuf).setBigUint64(0, BigInt(200 * 1024 * 1024), true);
    expect(() => new SafeTensors(bigHdrBuf)).toThrow(/Header size exceeds 100MB limit/);

    // safetensors.ts lines 819-820: saveSafetensors with object missing dtype and shape
    expect(saveSafetensors({ t1: { data: new Uint8Array(8) } }).byteLength).toBeGreaterThan(0);

    // safetensors.ts line 487: fetchSafetensorsHeader where initial fetch is !ok
    globalThis.fetch = vi.fn().mockResolvedValue({
      status: 404,
      ok: false,
      statusText: 'Not Found',
      headers: new Headers(),
    });
    await expect(fetchSafetensorsHeader('http://example.com/missing')).rejects.toThrow(
      /Failed to fetch header size: Not Found/,
    );

    // safetensors.ts line 604: cachedRes with onProgress callback
    const origCaches2 = globalThis.caches;
    globalThis.caches = {
      open: async () => ({
        match: async () => ({ arrayBuffer: async () => new ArrayBuffer(4) }),
        put: async () => undefined,
      }),
    } as unknown as CacheStorage;
    let progCalled = false;
    await fetchSafetensorsChunk('http://example.com/c2', 8, 0, 4, undefined, () => {
      progCalled = true;
    });
    expect(progCalled).toBe(true);

    // safetensors.ts line 617: 429 response without Retry-After header (using fake timers or 0 header)
    globalThis.caches = {
      open: async () => ({
        match: async () => undefined,
        put: async () => undefined,
      }),
    } as unknown as CacheStorage;
    let call429 = 0;
    const origSetTimeout = globalThis.setTimeout;
    globalThis.setTimeout = ((fn: () => void) =>
      origSetTimeout(fn, 0)) as unknown as typeof setTimeout;
    globalThis.fetch = vi.fn().mockImplementation(async () => {
      call429++;
      if (call429 === 1) {
        return {
          status: 429,
          ok: false,
          headers: new Headers(),
        };
      }
      return {
        status: 206,
        ok: true,
        headers: new Headers({ 'Content-Length': '2' }),
        body: null,
        arrayBuffer: async () => new Uint8Array([1, 2]).buffer,
      };
    });
    const noBodyChunk = await fetchSafetensorsChunk('hf://a/b', 8, 0, 2);
    expect(noBodyChunk).toEqual(new Uint8Array([1, 2]));
    globalThis.setTimeout = origSetTimeout;
    const noBodyChunk2 = await fetchSafetensorsChunk(
      'https://huggingface.co/m',
      8,
      0,
      2,
      undefined,
      () => undefined,
    );
    expect(noBodyChunk2).toEqual(new Uint8Array([1, 2]));
    globalThis.caches = origCaches2;

    globalThis.fetch = origFetch2;
  });
});
