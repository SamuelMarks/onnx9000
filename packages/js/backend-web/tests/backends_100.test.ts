import { Attribute, Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { describe, expect, it, vi } from 'vitest';
import { ONNXToMILConverter } from '../../coreml/src/converter.js';
import { emitModel } from '../../coreml/src/emitter.js';
import { Block, MILFunction, Operation, Program, Var } from '../../coreml/src/mil/ast.js';
import { applyCompression } from '../../coreml/src/mil/compression.js';
import { inferShapes, replaceOperation, replaceVarUsage } from '../../coreml/src/mil/rewriter.js';
import { MILDataType, ScalarType, TensorType } from '../../coreml/src/mil/types.js';
import { validateBlock, validateMILProgram } from '../../coreml/src/mil/validator.js';
import { OpenVinoExporter } from '../../openvino/src/exporter.js';
import { PolyfillMLGraphBuilder } from '../../webnn-polyfill/src/builder.js';
import { PolyfillMLContext } from '../../webnn-polyfill/src/context.js';
import { PolyfillMLGraph } from '../../webnn-polyfill/src/graph.js';
import { PolyfillMLTensor } from '../../webnn-polyfill/src/tensor.js';
import { GraphPartitioner } from '../src/partitioner.js';
import { loadProgressive } from '../src/progressive.js';
import { MatMulWebGPU, WebGPUProvider } from '../src/providers/webgpu/index.js';
import { WebGPUMemoryManager } from '../src/providers/webgpu/memory.js';
import { WebNNLayoutValidator } from '../src/providers/webnn/conformance_validator.js';
import { WebNNContextManager } from '../src/providers/webnn/context.js';
import { WebNNProvider } from '../src/providers/webnn/index.js';

describe('100% Coverage for JS Execution Backends', () => {
  it('covers backend-web partitioner, progressive, webgpu, and webnn branches', async () => {
    // 1. GraphPartitioner with empty providers
    const emptyPart = new GraphPartitioner([]);
    const gEmpty = new Graph('empty');
    expect(emptyPart.partition(gEmpty)[0]?.providerName).toBe('Unknown');
    const gNode = new Graph('one_node');
    gNode.addNode(new Node('Relu', ['x'], ['y'], {}, 'r'));
    expect(emptyPart.partition(gNode)[0]?.providerName).toBe('Unknown');

    // 2. ProgressiveSession
    const badSess = await loadProgressive('relative/path');
    await expect(badSess.run({})).rejects.toThrow('Progressive loading requires a valid HTTP URL');
    const goodSess = await loadProgressive('/model.onnx', { maxChunkSize: 512 });
    expect(await goodSess.run({})).toEqual({});
    expect(await goodSess.run({})).toEqual({}); // second call hits if (!this.isLoaded) false
    const defaultChunkSess = await loadProgressive('https://example.com/model.onnx');
    expect(await defaultChunkSess.run({})).toEqual({});

    // 3. MatMulWebGPU low sparsity & WebGPUProvider branches
    const smm = new MatMulWebGPU();
    vi.spyOn(
      smm as unknown as { calculateSparsity: () => number },
      'calculateSparsity',
    ).mockReturnValueOnce(0.3);
    const fakeSparse = new Tensor('w', [2, 2], 'float32') as Tensor & { format: string };
    fakeSparse.format = 'CSR';
    smm.execute([new Tensor('x', [2, 2], 'float32'), fakeSparse], {});

    const wgpu = new WebGPUProvider();
    expect(wgpu.createSparseBuffer({ format: 'CSR', values: new Float32Array(1) })).toBeNull();
    vi.stubGlobal('navigator', undefined);
    await expect(wgpu.initialize()).rejects.toThrow('WebGPU is not supported');
    vi.unstubAllGlobals();

    // 4. WebGPUMemoryManager
    const memMgr = new WebGPUMemoryManager(15 * 1024 * 1024);
    const gMem = new Graph('mem_g');
    gMem.addNode(new Node('Relu', ['x'], ['y'], {}, 'n1'));
    gMem.addNode(new Node('Relu', ['y'], ['z'], {}, 'n2'));
    expect(memMgr.checkAndPartition(gMem).length).toBe(2);
    expect(new WebGPUMemoryManager().checkAndPartition(new Graph('empty_mem')).length).toBe(0);

    // 5. WebNNLayoutValidator
    const gConvMissing = new Graph('c1');
    gConvMissing.addNode(new Node('Conv', ['x', 'w'], ['y'], {}, 'conv1'));
    expect(() => WebNNLayoutValidator.validateLayouts(gConvMissing)).toThrow(
      'missing a layout attribute',
    );

    const gConvBad = new Graph('c2');
    gConvBad.addNode(
      new Node(
        'Conv2D',
        ['x', 'w'],
        ['y'],
        { layout: new Attribute('layout', 'INT', 123) },
        'conv2',
      ),
    );
    expect(() => WebNNLayoutValidator.validateLayouts(gConvBad)).toThrow('invalid layout UNKNOWN');

    const gValid = new Graph('c3');
    gValid.addNode(
      new Node(
        'Conv',
        ['x', 'w'],
        ['y'],
        { layout: new Attribute('layout', 'STRING', 'NCHW') },
        'conv3',
      ),
    );
    gValid.addNode(new Node('Gemm', ['a', 'b'], ['c'], {}, 'gemm1'));
    gValid.addNode(new Node('MatMul', ['a', 'b'], ['c'], {}, 'matmul1'));
    gValid.addNode(
      new Node(
        'Gemm',
        ['a', 'b'],
        ['c'],
        {
          transA: new Attribute('transA', 'INT', 1),
          transB: new Attribute('transB', 'INT', 1),
        },
        'gemm2',
      ),
    );
    expect(WebNNLayoutValidator.validateLayouts(gValid)).toBe(true);

    // 6. WebNNContextManager window.MLGraphBuilder branch
    class FakeMLGraphBuilder {}
    vi.stubGlobal('window', { MLGraphBuilder: FakeMLGraphBuilder });
    vi.stubGlobal('MLGraphBuilder', FakeMLGraphBuilder);
    vi.stubGlobal('navigator', {
      ml: { createContext: vi.fn(async () => ({})) },
    });
    (WebNNContextManager as unknown as { instance: null }).instance = null;
    const ctxMgr = WebNNContextManager.getInstance();
    await ctxMgr.initialize();
    expect(ctxMgr.getBuilder()).toBeInstanceOf(FakeMLGraphBuilder);

    // 7. WebNNProvider non-Error rejection in compile and compute
    class FailingBuilder {
      input() {
        return {};
      }
      relu() {
        return {};
      }
      async build() {
        throw 'string_compile_error';
      }
    }
    vi.stubGlobal('window', undefined);
    vi.stubGlobal('MLGraphBuilder', FailingBuilder);
    (WebNNContextManager as unknown as { instance: null }).instance = null;
    const provider1 = new WebNNProvider();
    await provider1.initialize();
    const gExec = new Graph('exec_g');
    gExec.inputs.push(new ValueInfo('x', [1], 'float32'));
    gExec.outputs.push(new ValueInfo('y', [1], 'float32'));
    gExec.addNode(new Node('Relu', ['x'], ['y'], {}, 'r'));
    await expect(
      provider1.execute(gExec, {
        x: new Tensor('x', [1], 'float32', false, true, new Float32Array([1])),
      }),
    ).rejects.toThrow('string_compile_error');

    class ComputeFailBuilder {
      input() {
        return {};
      }
      relu() {
        return {};
      }
      async build() {
        return {};
      }
    }
    vi.stubGlobal('MLGraphBuilder', ComputeFailBuilder);
    vi.stubGlobal('navigator', {
      ml: {
        createContext: vi.fn(async () => ({
          compute: async () => {
            throw 'string_compute_error';
          },
        })),
      },
    });
    (WebNNContextManager as unknown as { instance: null }).instance = null;
    const provider2 = new WebNNProvider();
    await provider2.initialize();
    await expect(
      provider2.execute(gExec, {
        x: new Tensor('x', [1], 'float32', false, true, new Float32Array([1])),
      }),
    ).rejects.toThrow('string_compute_error');
    (WebNNContextManager as unknown as { instance: null }).instance = null;
    vi.unstubAllGlobals();
  });

  it('covers webnn-polyfill context, graph, tensor, and builder branches', async () => {
    // 1. PolyfillMLGraph without crypto
    const origCrypto = globalThis.crypto;
    vi.stubGlobal('crypto', undefined);
    const mlGraph = new PolyfillMLGraph(new Graph('g'));
    expect(mlGraph.compilationId.length).toBeGreaterThan(0);
    vi.stubGlobal('crypto', origCrypto);

    // 2. PolyfillMLTensor with unknown dataType and null internalBuffer in dispatch
    const tUnknown = new PolyfillMLTensor({
      dataType: 'unknown_type' as never,
      dimensions: [2],
    });
    tUnknown.destroy();

    const g = new Graph('poly_g');
    g.inputs.push(new ValueInfo('x', [1], 'float32'));
    g.outputs.push(new ValueInfo('y', [1], 'float32'));
    g.addNode(new Node('Identity', ['x'], ['y'], {}, 'id'));
    const pGraph = new PolyfillMLGraph(g);
    const pCtx = new PolyfillMLContext([]);
    (pCtx as unknown as { providers: unknown[] }).providers = [
      {
        name: 'Mock',
        initialize: async () => undefined,
        execute: async () => ({
          y: new Tensor('y', [1], 'float32', false, true, new Float32Array([1])),
        }),
      },
    ];
    await pCtx.dispatch(pGraph, { x: tUnknown }, { y: tUnknown });

    // 3. PolyfillMLGraphBuilder remaining option branches
    const builder = new PolyfillMLGraphBuilder(pCtx);
    const x = builder.input('x', { dataType: 'float32', dimensions: [2, 2, 2, 2] });
    const w = builder.input('w', { dataType: 'float32', dimensions: [2, 2, 1, 1] });
    const idx = builder.input('idx', { dataType: 'int32', dimensions: [2, 2] });
    const x1d = builder.input('x1d', { dataType: 'float32', dimensions: [4] });
    const x0d = builder.input('x0d', { dataType: 'float32', dimensions: [] });

    builder.leakyRelu(x);
    builder.elu(x);
    builder.gather(x, idx, { axis: 1 });
    builder.gather(x, idx);
    builder.gatherElements(x, idx, { axis: 1 });
    builder.gatherElements(x, idx);
    builder.scatterNd(idx, x, { shape: [2, 2] });
    builder.scatterNd(idx, x);
    builder.pad(x, [0, 0], { mode: 'constant', value: 1.0 });
    builder.pad(x, [0, 0]);
    builder.cast(x, 'unknown_dtype' as never);

    builder.conv2d(x, w, {
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [1, 1],
      groups: 1,
      autoPad: 'same-upper',
      bias: x1d,
    });
    builder.conv2d(x, w, { autoPad: 'other' as never });
    builder.convTranspose2d(x, w, {
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [1, 1],
      groups: 1,
      outputPadding: [0, 0],
      bias: x1d,
    });
    builder.maxPool2d(x, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [1, 1],
      roundingType: 'ceil',
    });
    builder.maxPool2d(x);
    builder.averagePool2d(x, { roundingType: 'ceil' });

    builder.reduceMean(x, { axes: [0], keepDimensions: false });
    builder.reduceMean(x, { keepDimensions: true });
    builder.argMax(x, { axes: [1], keepDimensions: false });
    builder.argMax(x);

    builder.batchNormalization(x, x1d, x1d, { scale: x1d, bias: x1d, epsilon: 1e-3 });
    builder.batchNormalization(x, x1d, x1d);
    builder.instanceNormalization(x1d, { scale: x1d, bias: x1d, epsilon: 1e-3 });
    builder.instanceNormalization(x1d);
    builder.layerNormalization(x0d, { scale: x1d, bias: x1d, axes: [0], epsilon: 1e-3 });
    builder.layerNormalization(x0d);
    builder.l2Normalization(x, { axes: [1] });
    builder.l2Normalization(x);

    builder.lstmCell(x, x, x, x, x, 4, {});
    builder.lstmCell(x, x, x, x, x, 4);
    builder.gruCell(x, x, x, x, 4, {});
    builder.gruCell(x, x, x, x, 4);
    builder.triangular(x, { upper: true });
    builder.triangular(x, { upper: false });
    builder.triangular(x);
  });

  it('covers coreml converter, emitter, compression, rewriter, and validator branches', () => {
    // 1. ONNXToMILConverter
    const g = new Graph('coreml_g');
    g.inputs.push(new ValueInfo('x', [1, 2, 4, 4], 'float32'));
    g.initializers.push('missing_init', 'w_fp16', 'w_i32', 'w_i64', 'w_bool', 'w_fp64');
    g.tensors.w_fp16 = new Tensor('w_fp16', [1], 'float16', true, false, new Uint16Array([1]));
    g.tensors.w_i32 = new Tensor('w_i32', [1], 'int32', true, false, new Int32Array([1]));
    g.tensors.w_i64 = new Tensor('w_i64', [1], 'int64', true, false, new BigInt64Array([1n]));
    g.tensors.w_bool = new Tensor('w_bool', [1], 'bool', true, false, new Uint8Array([1]));
    g.tensors.w_fp64 = new Tensor('w_fp64', [1], 'float64', true, false, new Float64Array([1]));

    g.addNode(new Node('Pad', ['x'], ['pad_out'], {}, 'pad1'));
    g.addNode(
      new Node(
        'Conv',
        ['pad_out', 'w_fp16'],
        ['conv1'],
        {
          group: new Attribute('group', 'INT', 2),
          auto_pad: new Attribute('auto_pad', 'STRING', 'SAME_UPPER'),
          pads: new Attribute('pads', 'INTS', [0, 0, 0, 0]),
        },
        'conv1',
      ),
    );
    g.addNode(
      new Node(
        'Conv',
        ['conv1', 'w_fp16'],
        ['conv2'],
        { auto_pad: new Attribute('auto_pad', 'STRING', 'VALID') },
        'conv2',
      ),
    );
    g.addNode(
      new Node(
        'Conv',
        ['conv2', 'w_fp16'],
        ['conv3'],
        { auto_pad: new Attribute('auto_pad', 'STRING', 'NOTSET') },
        'conv3',
      ),
    );
    g.addNode(new Node('BatchNormalization', ['conv3'], ['bn1'], {}, 'bn1'));
    g.addNode(
      new Node(
        'Resize',
        ['bn1'],
        ['res1'],
        {
          mode: new Attribute('mode', 'STRING', 'linear'),
          coordinate_transformation_mode: new Attribute(
            'coordinate_transformation_mode',
            'STRING',
            'align_corners',
          ),
        },
        'res1',
      ),
    );
    g.addNode(
      new Node(
        'Resize',
        ['res1'],
        ['res2'],
        {
          coordinate_transformation_mode: new Attribute(
            'coordinate_transformation_mode',
            'STRING',
            'asymmetric',
          ),
        },
        'res2',
      ),
    );
    g.addNode(new Node('Resize', ['res2'], ['res3'], {}, 'res_half_pixel'));
    g.addNode(
      new Node(
        'ReduceMean',
        ['res3'],
        ['red1'],
        { keepdims: new Attribute('keepdims', 'INT', 1), axis: new Attribute('axis', 'INT', 0) },
        'red1',
      ),
    );
    g.addNode(
      new Node(
        'LSTM',
        ['red1'],
        ['lstm1'],
        {
          direction: new Attribute('direction', 'STRING', 'BIDIRECTIONAL'),
          layout: new Attribute('layout', 'INT', 0),
        },
        'lstm1',
      ),
    );
    g.addNode(new Node('GRU', ['lstm1'], ['gru1'], {}, 'gru1'));
    g.addNode(
      new Node(
        'Loop',
        ['gru1', 'unmapped_in'],
        ['loop1'],
        { body: new Attribute('body', 'GRAPH', new Graph('sub')) },
        'loop1',
      ),
    );
    g.outputs.push(new ValueInfo('loop1', [1], 'float32'));

    const conv = new ONNXToMILConverter(g, { dynamicBatching: true });
    const prog = conv.convert();
    expect(prog).toBeDefined();

    // Missing output throws Onnx9000Error
    const gBadOut = new Graph('bad_out');
    gBadOut.outputs.push(new ValueInfo('missing_out', [1], 'float32'));
    expect(() => new ONNXToMILConverter(gBadOut).convert()).toThrow('not found in MIL graph');

    // Unsupported op with telemetry enabled
    process.env.ONNX9000_TELEMETRY = '1';
    const gUnsupported = new Graph('unsup');
    gUnsupported.addNode(new Node('CustomUnsupportedOp', [], ['y'], {}, 'u'));
    expect(() => new ONNXToMILConverter(gUnsupported).convert()).toThrow(
      'Operation is completely unsupported',
    );
    delete process.env.ONNX9000_TELEMETRY;

    // 2. emitter.ts
    const modelBytes = emitModel({
      specificationVersion: 7,
      description: {
        input: [
          { name: 'i64', shortDescription: 'desc', type: { int64Type: {} } },
          { name: 'f64', type: { doubleType: {} } },
          { name: 'str', type: { stringType: {} } },
          {
            name: 'img',
            type: {
              imageType: { width: 224, height: 224, colorSpace: 20 },
            },
          },
          {
            name: 'arr',
            type: {
              multiArrayType: { shape: [1, 3], dataType: 65568 },
            },
          },
        ],
        output: [],
        metadata: {
          shortDescription: 'short',
          versionString: '1.0',
          author: 'onnx9000',
          license: 'Apache-2.0',
          creatorDefined: { k1: 'v1' },
        },
      },
      neuralNetwork: {
        layers: [{ name: 'l1', input: ['i64'], output: ['o1'] }],
      },
    });
    expect(modelBytes.byteLength).toBeGreaterThan(0);

    const mlModelBytes = emitModel({
      specificationVersion: 7,
      mlProgram: {
        version: 1,
        functions: {
          main: { inputs: [{ name: 'i64' }] },
        },
      },
    });
    expect(mlModelBytes.byteLength).toBeGreaterThan(0);
    const mlEmptyFuncs = emitModel({
      specificationVersion: 7,
      mlProgram: { version: 1 },
    });
    expect(mlEmptyFuncs.byteLength).toBeGreaterThan(0);

    // 3. mil/compression.ts
    const compBlock = new Block('b0');
    const vConst1 = new Var('w1', new TensorType(MILDataType.FLOAT32, [4, 4]));
    const vConst2 = new Var('w2', new TensorType(MILDataType.FLOAT32, [4, 4]));
    const vConst3 = new Var('w3_unmapped', new TensorType(MILDataType.FLOAT32, [4, 4]));
    compBlock.operations.push(new Operation('const', {}, [vConst1], {}));
    compBlock.operations.push(new Operation('const', {}, [vConst2], {}));
    compBlock.operations.push(new Operation('const', {}, [vConst3], {}));
    compBlock.operations.push(new Operation('quantize_linear', {}, [], {}));
    compBlock.operations.push(new Operation('read_state', {}, [], {}));
    applyCompression(compBlock, {
      mode: 'mixed',
      mixedPrecisionConfig: { w1: 'w8a16', w2: 'w4a16' },
      multiBitrate: true,
      kvCacheQuantization: true,
    });
    const palBlock = new Block('pal');
    palBlock.operations.push(new Operation('const', {}, [vConst1], {}));
    applyCompression(palBlock, { mode: 'palettization' });
    const sparseBlock = new Block('sp');
    sparseBlock.operations.push(new Operation('const', {}, [vConst1], {}));
    applyCompression(sparseBlock, { mode: 'sparse' });
    expect(compBlock.operations[3]?.attributes.ane_hint_mapped_qlinear).toBe(true);
    expect(compBlock.operations[4]?.attributes.kv_cache_quantized).toBe('int4');

    // 4. mil/rewriter.ts and mil/validator.ts
    const rwBlock = new Block('rw');
    const vIn = new Var('in1', new TensorType(MILDataType.FLOAT32, [2, 3]));
    const vIn2 = new Var('in2', new TensorType(MILDataType.FLOAT32, [1, 2, 3]));
    const vScalar = new Var('sc', new ScalarType(MILDataType.FLOAT32));
    rwBlock.inputs.push(vIn, vIn2, vScalar);

    const vAddOut = new Var('add_out', new TensorType(MILDataType.FLOAT32, []));
    const opAdd = new Operation('add', { x: vIn2, y: vIn }, [vAddOut], {});
    const vSubOut = new Var('sub_out', new TensorType(MILDataType.FLOAT32, []));
    const opSubUnmapped = new Operation('sub', { x: vScalar, y: vScalar }, [vSubOut], {});
    const vConst = new Var('c_var', new TensorType(MILDataType.FLOAT32, [2, 3]));
    const opConst = new Operation('const', {}, [vConst], {});
    const vUnk1 = new Var('u1', new TensorType(MILDataType.FLOAT32, [5]));
    const opUnk1 = new Operation('custom1', { x: [vIn, vIn2], y: vIn }, [vUnk1], {});
    const vUnk2 = new Var('u2', new TensorType(MILDataType.FLOAT32, []));
    const opUnk2 = new Operation('custom2', { arr: [vIn], x: vIn }, [vUnk2], {});

    rwBlock.operations.push(opAdd, opSubUnmapped, opConst, opUnk1, opUnk2);
    rwBlock.outputs.push(vIn);

    const vReplacement = new Var('in1_rep', new TensorType(MILDataType.FLOAT32, [2, 3]));
    replaceVarUsage(rwBlock, vIn, vReplacement);
    rwBlock.inputs[0] = vReplacement;
    inferShapes(rwBlock);
    expect((vUnk2.type as TensorType).shape).toEqual([2, 3]);

    replaceOperation(rwBlock, opUnk1, [opUnk1]);
    expect(() => replaceOperation(rwBlock, new Operation('missing', {}, [], {}), [])).toThrow(
      'not found in block',
    );

    const validFn = new MILFunction('main', [], []);
    validFn.addBlock(rwBlock);
    const validProg = new Program();
    validProg.addFunction(validFn);
    expect(validateMILProgram(validProg)).toBe(true);

    const badBlock1 = new Block('bad1');
    badBlock1.operations.push(
      new Operation(
        'add',
        { x: [new Var('missing_arr', new ScalarType(MILDataType.FLOAT32))] },
        [],
        {},
      ),
    );
    expect(() => validateBlock(badBlock1)).toThrow('Operation input missing_arr is not available');

    const badBlock2 = new Block('bad2');
    badBlock2.outputs.push(new Var('missing_block_out', new ScalarType(MILDataType.FLOAT32)));
    expect(() => validateBlock(badBlock2)).toThrow(
      'Block output missing_block_out is not produced',
    );
  });

  it('covers openvino exporter.ts remaining branches', () => {
    const g = new Graph('');
    g.inputs.push(
      new ValueInfo('unused_in', [1], 'float32'),
      new ValueInfo('x', [-1, 'batch', 4, 4], 'float32'),
      new ValueInfo('scalar_in', [], 'float32'),
    );
    g.outputs.push(
      new ValueInfo('unmapped_out', [1], 'float32'),
      new ValueInfo('final_out', [1], 'float32'),
    );

    const f32Data = new Float32Array([0.0, 1e-40, 1e38, 1.5]);
    g.initializers.push('missing_init', 'w_fp32', 'w_fp32_dup', 'w_empty', 'w_scalar');
    g.tensors.w_fp32 = new Tensor('w_fp32', [1, 1, 2, 2], 'float32', true, false, f32Data);
    g.tensors.w_fp32_dup = new Tensor('w_fp32_dup', [1, 1, 2, 2], 'float32', true, false, f32Data);
    g.tensors.w_empty = new Tensor('w_empty', [1], 'float32', true, false, null);
    g.tensors.w_scalar = new Tensor(
      'w_scalar',
      [],
      'float32',
      true,
      false,
      new Float32Array([1.0]),
    );
    for (const dt of [
      'float64',
      'float16',
      'bfloat16',
      'int64',
      'int32',
      'int16',
      'int8',
      'uint64',
      'uint32',
      'uint16',
      'uint8',
      'bool',
    ] as const) {
      g.initializers.push(`w_${dt}`);
      g.tensors[`w_${dt}`] = new Tensor(`w_${dt}`, [1], dt, true, false, null);
    }

    // Add nodes covering all handlers in OpenVinoExporter
    g.addNode(
      new Node(
        'Conv',
        ['x', 'w_fp32', 'w_scalar'],
        ['conv1'],
        {
          strides: new Attribute('strides', 'INTS', [1, 1]),
          dilations: new Attribute('dilations', 'INTS', [1, 1]),
          pads: new Attribute('pads', 'INTS', [0, 0, 0, 0]),
          auto_pad: new Attribute('auto_pad', 'STRING', 'SAME_UPPER'),
        },
        '',
      ),
    );
    g.addNode(
      new Node(
        'Conv',
        ['conv1', 'w_fp32', ''],
        ['conv2'],
        {
          pads: new Attribute('pads', 'INTS', [0, 0, 0, 0, 0, 0]),
          auto_pad: new Attribute('auto_pad', 'STRING', 'OTHER'),
        },
        'conv_6d',
      ),
    );
    g.addNode(
      new Node(
        'AveragePool',
        ['conv2'],
        ['pool1'],
        {
          kernel_shape: new Attribute('kernel_shape', 'INTS', [2, 2]),
          strides: new Attribute('strides', 'INTS', [1, 1]),
          pads: new Attribute('pads', 'INTS', [0, 0, 0, 0]),
          auto_pad: new Attribute('auto_pad', 'STRING', 'VALID'),
          count_include_pad: new Attribute('count_include_pad', 'INT', 1),
        },
        'pool_4d',
      ),
    );
    g.addNode(
      new Node(
        'AveragePool',
        ['pool1'],
        ['pool2'],
        {
          pads: new Attribute('pads', 'INTS', [0, 0, 0, 0, 0, 0]),
          auto_pad: new Attribute('auto_pad', 'STRING', 'OTHER'),
          count_include_pad: new Attribute('count_include_pad', 'INT', 0),
        },
        'pool_6d',
      ),
    );
    g.addNode(
      new Node(
        'Gelu',
        ['pool2'],
        ['gelu1'],
        { approximate: new Attribute('approximate', 'STRING', 'tanh') },
        'gelu_tanh',
      ),
    );
    g.addNode(new Node('Gelu', ['gelu1'], ['gelu2'], {}, 'gelu_erf'));
    g.addNode(
      new Node(
        'Pad',
        ['gelu2'],
        ['pad1'],
        {
          pads: new Attribute('pads', 'INTS', [0, 0, 1, 1]),
          value: new Attribute('value', 'FLOAT', 2.0),
        },
        '',
      ),
    );
    g.addNode(new Node('Pad', ['pad1', 'w_scalar'], ['pad2'], {}, ''));
    g.addNode(new Node('Pad', ['pad2', 'w_scalar', 'w_scalar'], ['pad3'], {}, 'pad_3in'));
    g.addNode(
      new Node(
        'Gather',
        ['pad3', 'w_scalar'],
        ['gath1'],
        {
          batch_dims: new Attribute('batch_dims', 'INT', 0),
          axis: new Attribute('axis', 'INT', 0),
        },
        '',
      ),
    );
    g.addNode(new Node('Slice', ['gath1'], ['slice1'], {}, 'slice1'));
    g.addNode(
      new Node(
        'ReduceMean',
        ['slice1'],
        ['red1'],
        {
          keepdims: new Attribute('keepdims', 'INT', 1),
          axes: new Attribute('axes', 'INTS', [0]),
        },
        '',
      ),
    );
    g.addNode(
      new Node(
        'ReduceMean',
        ['red1'],
        ['red2'],
        { keepdims: new Attribute('keepdims', 'INT', 0) },
        'red2',
      ),
    );
    g.addNode(
      new Node(
        'ArgMax',
        ['red2'],
        ['arg1'],
        {
          keepdims: new Attribute('keepdims', 'INT', 1),
          axis: new Attribute('axis', 'INT', 0),
        },
        'arg1',
      ),
    );
    g.addNode(
      new Node(
        'ArgMin',
        ['arg1'],
        ['arg2'],
        { keepdims: new Attribute('keepdims', 'INT', 0) },
        'arg2',
      ),
    );
    g.addNode(new Node('Resize', ['arg2'], ['res1'], {}, 'res1'));
    g.addNode(new Node('SpaceToDepth', ['res1'], ['s2d1'], {}, 's2d1'));
    g.addNode(
      new Node(
        'NonMaxSuppression',
        ['s2d1'],
        ['nms1'],
        { center_point_box: new Attribute('center_point_box', 'INT', 1) },
        'nms1',
      ),
    );
    g.addNode(new Node('NonMaxSuppression', ['nms1'], ['nms2'], {}, 'nms2'));
    g.addNode(
      new Node(
        'RoiAlign',
        ['nms2'],
        ['roi1'],
        { mode: new Attribute('mode', 'STRING', 'max') },
        'roi1',
      ),
    );
    g.addNode(new Node('RoiAlign', ['roi1'], ['roi2'], {}, 'roi2'));
    g.addNode(new Node('QuantizeLinear', ['roi2'], ['ql1'], {}, 'ql1'));
    g.addNode(new Node('Einsum', ['ql1'], ['ein1'], {}, 'ein1'));
    g.addNode(new Node('LayerNormalization', ['ein1'], ['ln1'], {}, 'ln1'));
    g.addNode(new Node('InstanceNormalization', ['ln1'], ['in1'], {}, 'in1'));
    g.addNode(new Node('LpNormalization', ['in1'], ['lp1'], {}, 'lp1'));
    g.addNode(new Node('BatchNormalization', ['lp1'], ['bn1'], {}, 'bn1'));
    const subG = new Graph('sub');
    g.addNode(
      new Node(
        'If',
        ['bn1'],
        ['if1'],
        {
          then_branch: new Attribute('then_branch', 'GRAPH', subG),
          else_branch: new Attribute('else_branch', 'GRAPH', subG),
        },
        'if1',
      ),
    );
    g.addNode(new Node('If', ['if1'], ['if2'], {}, 'if2'));
    g.addNode(
      new Node('Loop', ['if2'], ['loop1'], { body: new Attribute('body', 'GRAPH', subG) }, 'loop1'),
    );
    g.addNode(new Node('Loop', ['loop1'], ['loop2'], {}, 'loop2'));
    g.addNode(
      new Node(
        'Cast',
        ['loop2'],
        ['cast1'],
        { to: new Attribute('to', 'STRING', 'float32') },
        'cast1',
      ),
    );
    g.addNode(new Node('Cast', ['cast1'], ['cast2'], {}, 'cast2'));
    g.addNode(
      new Node(
        'GridSample',
        ['cast2'],
        ['gs1'],
        { align_corners: new Attribute('align_corners', 'INT', 1) },
        'gs1',
      ),
    );
    g.addNode(
      new Node(
        'GridSample',
        ['gs1'],
        ['gs2'],
        { align_corners: new Attribute('align_corners', 'INT', 0) },
        'gs2',
      ),
    );
    g.addNode(new Node('Size', ['gs2'], ['size1'], {}, ''));
    g.addNode(new Node('Size', ['unmapped_size_in'], ['size2'], {}, 'size2'));
    g.addNode(new Node('Size', [], ['size3'], {}, 'size3'));
    g.addNode(new Node('Flatten', ['size1'], ['flat1'], {}, ''));
    g.addNode(
      new Node(
        'Transpose',
        ['flat1'],
        ['tr1'],
        { perm: new Attribute('perm', 'INTS', [1, 0]) },
        '',
      ),
    );
    g.addNode(new Node('Transpose', ['tr1'], ['tr2'], {}, 'tr2'));
    g.addNode(
      new Node('GatherElements', ['tr2'], ['ge1'], { axis: new Attribute('axis', 'INT', 0) }, ''),
    );
    g.addNode(new Node('GatherElements', ['ge1'], ['ge2'], {}, 'ge2'));
    g.addNode(
      new Node(
        'ConstantOfShape',
        ['ge2'],
        ['cos1'],
        {
          value: new Attribute(
            'value',
            'TENSOR',
            new Tensor('t1', [1], 'float32', true, false, new Float32Array([1.0])),
          ),
        },
        '',
      ),
    );
    g.addNode(
      new Node(
        'ConstantOfShape',
        ['cos1'],
        ['cos2'],
        {
          value: new Attribute(
            'value',
            'TENSOR',
            new Tensor('t2', [1], 'int64', true, false, new BigInt64Array([2n])),
          ),
        },
        'cos2',
      ),
    );
    g.addNode(
      new Node(
        'ConstantOfShape',
        ['cos2'],
        ['cos3'],
        {
          value: new Attribute(
            'value',
            'TENSOR',
            new Tensor('t3', [1], 'int32', true, false, new Int32Array([3])),
          ),
        },
        'cos3',
      ),
    );
    g.addNode(
      new Node(
        'ConstantOfShape',
        ['cos3'],
        ['cos4'],
        {
          value: new Attribute('value', 'TENSOR', new Tensor('t4', [1], 'bool', true, false, null)),
        },
        'cos4',
      ),
    );
    g.addNode(new Node('Softmax', ['cos4'], ['soft_out'], {}, 'soft_default_axis'));
    g.addNode(
      new Node(
        'MatMul',
        ['soft_out', 'soft_out'],
        ['mm_out1'],
        {
          transA: new Attribute('transA', 'INT', 1),
          transB: new Attribute('transB', 'INT', 1),
        },
        'mm_trans',
      ),
    );
    g.addNode(new Node('MatMul', ['mm_out1', 'mm_out1'], ['mm_out'], {}, 'mm_no_trans'));
    g.addNode(new Node('Add', ['mm_out', 'mm_out'], ['final_out'], {}, 'dup_edge'));
    g.addNode(new Node('Identity', ['scalar_in', ''], [], {}, 'empty_io'));

    const exp = new OpenVinoExporter(g, { compressToFp16: true, clampDynamic: true });
    exp.emitShape([1, 2]);
    (exp as unknown as { addEdge: (a: number, b: number, c: number, d: number) => void }).addEdge(
      999,
      0,
      998,
      0,
    );
    (exp as unknown as { addEdge: (a: number, b: number, c: number, d: number) => void }).addEdge(
      999,
      0,
      998,
      0,
    );
    const res = exp.export();
    expect(res.xml).toContain('<net');

    const exp2 = new OpenVinoExporter(g, { compressToFp16: false, clampDynamic: false });
    exp2.emitShape(['-1']);
    exp2.emitDynamicConst('c_i32', [1], [], 'int32');
    exp2.emitDynamicConst('c_i32_dup', [1], [], 'int32');
    expect(() => exp2.emitDynamicConst('c_bad', [1], [1], 'bool')).toThrow(
      'Unsupported dtype for dynamic const',
    );
    expect(() =>
      (exp2 as unknown as { mapDtype: (d: string) => string }).mapDtype('invalid_dtype'),
    ).toThrow('Unsupported dtype for OpenVINO');

    const gMissingIn = new Graph('missing_in');
    gMissingIn.addNode(new Node('Relu', ['missing_ptr'], ['y'], {}, ''));
    expect(() => new OpenVinoExporter(gMissingIn).export()).toThrow('Missing input pointer');
  });
});
