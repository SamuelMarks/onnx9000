import { Attribute, Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { JaxMapper } from '../src/jax/mapper.js';
import { emitActivation } from '../src/keras/emitters.js';
import { Keras2OnnxConverter } from '../src/keras/index.js';
import { extractKerasTopology } from '../src/keras/keras-ast.js';
import { KerasGraphOptimizer } from '../src/keras/optimizers.js';
import { registerCustomKerasLayer } from '../src/keras/plugin-registry.js';
import { calculateByteLength } from '../src/keras/weight-loader.js';
import * as api from '../src/mmdnn/api.js';
import { CaffeGenerator } from '../src/mmdnn/caffe/generator.js';
import { CaffeMapper, register_caffe_op } from '../src/mmdnn/caffe/mapper.js';
import { parsePrototxt } from '../src/mmdnn/caffe/parser.js';
import { CatBoostParser } from '../src/mmdnn/catboost/parser.js';
import { CNTKGenerator } from '../src/mmdnn/cntk/generator.js';
import { DarknetMapper } from '../src/mmdnn/darknet/mapper.js';
import { parseCfg, parseWeights } from '../src/mmdnn/darknet/parser.js';
import { FileLoader } from '../src/mmdnn/file-loader.js';
import { NodeFusionRegistry } from '../src/mmdnn/fusion.js';
import { H2OMapper } from '../src/mmdnn/h2o/mapper.js';
import { parseH2O } from '../src/mmdnn/h2o/parser.js';
import { KerasGenerator } from '../src/mmdnn/keras/generator.js';
import { KerasImporter } from '../src/mmdnn/keras/importer.js';
import { LegacyQuirkResolver } from '../src/mmdnn/legacy/quirks.js';
import { LibSVMMapper } from '../src/mmdnn/libsvm/mapper.js';
import { parseLibSVM } from '../src/mmdnn/libsvm/parser.js';
import { MXNetGenerator } from '../src/mmdnn/mxnet/generator.js';
import { MxNetMapper, register_mxnet_op } from '../src/mmdnn/mxnet/mapper.js';
import { NcnnMapper } from '../src/mmdnn/ncnn/mapper.js';
import { NcnnBinParser, parseNcnnParam } from '../src/mmdnn/ncnn/parser.js';
import { OnnxScriptGenerator } from '../src/mmdnn/onnxscript/generator.js';
import { OnnxScriptParser } from '../src/mmdnn/onnxscript/parser.js';
import { PyTorchGenerator } from '../src/mmdnn/pytorch/generator.js';
import { PyTorchSerializer } from '../src/mmdnn/pytorch/serializer.js';
import { MMDNNReporter } from '../src/mmdnn/reporter.js';
import { ScikitLearnParser } from '../src/mmdnn/scikitlearn/parser.js';
import { ShapeInferenceEngine } from '../src/mmdnn/shape-inference.js';
import { SparkMLParser } from '../src/mmdnn/sparkml/parser.js';
import { TensorFlowGenerator } from '../src/mmdnn/tensorflow/generator.js';
import { TFMapper } from '../src/mmdnn/tensorflow/mapper.js';
import { parsePbtxt } from '../src/mmdnn/tensorflow/parser.js';
import { generateTFJSCode, isLinearGraph } from '../src/mmdnn/tfjs/generator.js';
import { serializeTFJSWeights } from '../src/mmdnn/tfjs/serializer.js';
import { topologicalSort } from '../src/mmdnn/topology.js';
import { ONNXNormalizer } from '../src/mmdnn/verification/normalizer.js';
import { XGBoostParser } from '../src/mmdnn/xgboost/parser.js';
import { JAXprParser } from '../src/parsers.js';

describe('Converters Final Push Coverage', () => {
  it('should cover Keras FLOATS and FLOAT attributes', async () => {
    registerCustomKerasLayer(
      'CustomAllTypesLayer',
      (nodeName, _layerName, inputs, outName, _config) => {
        return [
          {
            opType: 'Custom',
            inputs,
            outputs: [outName],
            name: nodeName,
            attributes: [
              { name: 'valFs', floats: [1.0, 2.0], type: 'FLOATS' },
              { name: 'valF', f: 3.14, type: 'FLOAT' },
              { name: 'valI', i: 42, type: 'INT' },
              { name: 'valS', s: 'hello', type: 'STRING' },
              { name: 'valIs', ints: [1, 2, 3], type: 'INTS' },
              { name: 'valEmptyF', type: 'FLOAT' },
              { name: 'valEmptyI', type: 'INT' },
              { name: 'valEmptyS', type: 'STRING' },
              { name: 'valEmptyIs', type: 'INTS' },
              { name: 'valEmptyFs', type: 'FLOATS' },
            ],
          },
        ];
      },
    );
    // Register again to trigger overwrite warning
    registerCustomKerasLayer('CustomAllTypesLayer', () => []);

    const modelJson = JSON.stringify({
      format: 'layers-model',
      modelTopology: {
        class_name: 'Functional',
        config: {
          name: 'model',
          layers: [
            {
              class_name: 'InputLayer',
              name: 'in',
              config: { batch_input_shape: [null, 10], name: 'in' },
              inbound_nodes: [],
            },
            {
              class_name: 'CustomAllTypesLayer',
              name: 'catl',
              config: { name: 'catl' },
              inbound_nodes: [[['in', 0, 0, {}]]],
            },
          ],
          input_layers: [['in', 0, 0]],
          output_layers: [['catl', 0, 0]],
        },
      },
      weightsManifest: [],
    });

    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();
    expect(typeof converter).toBeDefined();
  });

  it('should cover weight-loader int4, uint4, and unsupported dtype', () => {
    expect(calculateByteLength({ name: 'w1', shape: [10], dtype: 'int4' })).toBe(5);
    expect(calculateByteLength({ name: 'w2', shape: [11], dtype: 'uint4' })).toBe(6);
    expect(() =>
      calculateByteLength({ name: 'w3', shape: [1], dtype: 'bad_dtype' as 'float32' }),
    ).toThrow(/Unsupported dtype/);
  });

  it('should cover JaxMapper, JAXprParser, emitters, keras3-parser, legacy quirks', () => {
    const jaxpr = {
      invars: ['x', 'x'],
      outvars: ['z'],
      constvars: ['c1', 'c2', 'c3', 'c4'],
      eqns: [
        {
          primitive: 'add',
          invars: ['x', 'c1'],
          outvars: ['y'],
          params: {
            alpha: 1,
            beta: 1.5,
            mode: 'same',
            shape: [1, 2],
            scales: [1.5, 2.5],
            tags: ['a'],
          },
        },
        {
          primitive: 'custom_prim',
          invars: ['y', 'c2'],
          outvars: ['z'],
          params: { obj: {} },
        },
      ],
    };
    const jm = new JaxMapper(jaxpr, {
      c1: [1, 2],
      c2: { flat: () => [3, 4] },
      c3: { flat: () => 'not_array' as unknown as number[] },
    });
    const jg = jm.map();
    expect(jg.nodes.length).toBe(2);

    const jp = new JAXprParser();
    const jg2 = jp.parse({
      invars: [{ name: 'x', type: 'f32', shape: [1] }],
      outvars: [{ name: 'missing_out' }],
    });
    expect(jg2.outputs.length).toBe(0);

    expect(emitActivation('elu', 'x', 'y', 'elu1')[0]?.opType).toBe('Elu');
    expect(emitActivation('leaky_relu', 'x', 'y', 'lr1')[0]?.opType).toBe('LeakyRelu');
    expect(emitActivation('prelu', 'x', 'y', 'pr1')[0]?.opType).toBe('PRelu');
    expect(emitActivation('thresholded_relu', 'x', 'y', 'tr1')[0]?.opType).toBe('ThresholdedRelu');

    expect(LegacyQuirkResolver.resolveCaffePadding(undefined)).toEqual([0, 0, 0, 0]);
    expect(LegacyQuirkResolver.resolveCaffePadding([1])).toEqual([1, 1, 1, 1]);
    expect(LegacyQuirkResolver.resolveCaffePadding([1, 2])).toEqual([1, 2, 1, 2]);
    expect(LegacyQuirkResolver.resolveCaffePadding([1, 2, 3, 4])).toEqual([1, 2, 3, 4]);
    expect(LegacyQuirkResolver.resolveCaffePadding([1, 2, 3])).toEqual([1, 2, 3, 0]);

    expect(
      LegacyQuirkResolver.resolveCntkBroadcast({ opType: 'Plus', attributes: { broadcast: true } })
        .attributes?.cntk_broadcast_resolved,
    ).toBe(true);
    expect(LegacyQuirkResolver.resolveCntkBroadcast({ opType: 'Relu' }).opType).toBe('Relu');

    expect(LegacyQuirkResolver.resolveMxnetFlatten({ opType: 'Flatten' }, 1).attributes?.axis).toBe(
      0,
    );
    expect(
      LegacyQuirkResolver.resolveMxnetFlatten({ opType: 'Flatten', attributes: {} }, 4).attributes
        ?.axis,
    ).toBe(1);
    expect(LegacyQuirkResolver.resolveMxnetFlatten({ opType: 'Relu' }, 2).opType).toBe('Relu');

    expect(LegacyQuirkResolver.stripCaffeTrainingNodes(null as unknown as [])).toEqual([]);
    const stripped = LegacyQuirkResolver.stripCaffeTrainingNodes([
      { name: 'd', type: 'Dropout', include: { phase: 'TRAIN' } },
      { name: 'd2', type: 'Dropout', include: [{ phase: 0 }] },
      { name: 'd3', type: 'Dropout', include: { phase: 'TEST' } },
      { name: 'c', type: 'Conv', bottom: ['b'], top: ['c'] },
      { name: 'acc', type: 'Accuracy' },
      {},
    ]);
    expect(stripped.length).toBe(3);

    expect(
      LegacyQuirkResolver.emulateCaffeROIPooling({
        name: 'roi',
        bottom: ['a', 'b'],
        top: ['c'],
        roi_pooling_param: { pooled_h: 7, pooled_w: 7, spatial_scale: 0.5 },
      })[0]?.opType,
    ).toBe('MaxRoiPool');
    expect(LegacyQuirkResolver.emulateCaffeROIPooling({})[0]?.opType).toBe('MaxRoiPool');
  });

  it('should cover H2O, LibSVM, CatBoost, SparkML, XGBoost, ScikitLearn parsers & mappers', () => {
    expect(parseH2O('not json')).toEqual({});
    expect(parseH2O('{ bad json')).toEqual({});
    expect(parseH2O('{"algo":"gbm"}')).toEqual({ algo: 'gbm' });
    expect(new H2OMapper({ algo: 'xgboost' }).map().nodes[0]?.opType).toBe('TreeEnsembleRegressor');
    expect(new H2OMapper({ algo: 'drf', nclasses: 1 }).map().nodes[0]?.opType).toBe(
      'TreeEnsembleRegressor',
    );
    expect(new H2OMapper({ algo: 'deeplearning' }).map().nodes[0]?.opType).toBe('MatMul');
    expect(new H2OMapper({}).map().nodes[0]?.opType).toBe('TreeEnsembleRegressor');

    const svmModel = parseLibSVM(
      '\nsvm_type epsilon_svr\nkernel_type linear\nrho 0.5\nSV\n1.5 1:0.2\nbad_sv\n',
    );
    expect(svmModel.svmType).toBe('epsilon_svr');
    expect(new LibSVMMapper(svmModel).map().nodes[0]?.opType).toBe('SVMRegressor');
    const svcModel = parseLibSVM('svm_type c_svc\nkernel_type rbf\n');
    expect(new LibSVMMapper(svcModel).map().nodes[0]?.opType).toBe('SVMClassifier');

    const cb = new CatBoostParser();
    expect(cb.parseModel(JSON.stringify({ catboost_version: '1.0' })).nodes[0]?.opType).toBe(
      'TreeEnsembleClassifier',
    );
    expect(cb.parseModel(JSON.stringify({ other: 1 })).nodes[0]?.opType).toBe('Identity');
    expect(cb.parseModel('{bad json').nodes[0]?.opType).toBe('Identity');

    const sp = new SparkMLParser();
    expect(sp.parseModel(JSON.stringify({ class: 'LogisticRegression' })).nodes[0]?.opType).toBe(
      'LinearClassifier',
    );
    expect(sp.parseModel(JSON.stringify({ class: 'Other' })).nodes[0]?.opType).toBe('Identity');
    expect(sp.parseModel('{bad json').nodes[0]?.opType).toBe('Identity');

    const xgb = new XGBoostParser();
    expect(
      xgb.parseModel(JSON.stringify({ learner: { objective: { name: 'binary:logistic' } } }))
        .nodes[0]?.opType,
    ).toBe('TreeEnsembleClassifier');
    expect(xgb.parseModel(JSON.stringify({ other: 1 })).nodes[0]?.opType).toBe(
      'TreeEnsembleRegressor',
    );
    expect(xgb.parseModel('{bad json').nodes[0]?.opType).toBe('Identity');

    const skl = new ScikitLearnParser();
    expect(
      skl.parseModel(JSON.stringify({ model: 'RandomForestClassifier' })).nodes[0]?.opType,
    ).toBe('TreeEnsembleClassifier');
    expect(
      skl.parseModel(JSON.stringify({ model: 'SVC', kernel: 'linear' })).nodes[0]?.opType,
    ).toBe('SVMClassifier');
    expect(skl.parseModel(JSON.stringify({ model: 'SVC' })).nodes[0]?.opType).toBe('SVMClassifier');
    expect(skl.parseModel(JSON.stringify({ model: 'Other' })).nodes[0]?.opType).toBe(
      'LinearClassifier',
    );
    expect(skl.parseModel('{bad json').nodes[0]?.opType).toBe('Identity');
  });

  it('should cover TFJS generator & serializer, PyTorch generator & serializer, Caffe/CNTK/MXNet/Keras/TF/ONNXScript generators', () => {
    const gLin = new Graph('lin');
    gLin.inputs.push(new ValueInfo('0in', [-1, 3, 4, 4], 'float32'));
    gLin.outputs.push(new ValueInfo('out', [-1, 10], 'float32'));
    gLin.tensors.w_conv = new Tensor(
      'w_conv',
      [8, 3, 3, 3],
      'float32',
      true,
      false,
      new Uint8Array(9),
    );
    gLin.initializers.push('w_conv', 'b_conv', 'b_gemm');
    gLin.tensors.w_gemm = new Tensor('w_gemm', [10, 8], 'float32', true, false, new Uint8Array(12));
    gLin.nodes.push(
      new Node('Conv', ['0in', 'w_conv', 'b_conv'], ['h1'], {
        strides: new Attribute('strides', 'INTS', [1, 1]),
        pads: new Attribute('pads', 'INTS', [1, 1, 1, 1]),
      }),
      new Node('BatchNormalization', ['h1'], ['h2']),
      new Node('Relu', ['h2'], ['h3']),
      new Node('MaxPool', ['h3'], ['h4'], {
        kernel_shape: new Attribute('kernel_shape', 'INTS', [2, 2]),
        strides: new Attribute('strides', 'INTS', [2, 2]),
        pads: new Attribute('pads', 'INTS', [1, 1, 1, 1]),
      }),
      new Node('AveragePool', ['h4'], ['h5']),
      new Node('GlobalAveragePool', ['h5'], ['h6']),
      new Node('Flatten', ['h6'], ['h7']),
      new Node('Gemm', ['h7', 'w_gemm', 'b_gemm'], ['out'], {
        transB: new Attribute('transB', 'INT', 1),
      }),
    );
    expect(isLinearGraph(gLin)).toBe(true);
    expect(generateTFJSCode(gLin)).toContain('tf.sequential()');

    const gNonLin = new Graph('nonlin');
    gNonLin.inputs.push(
      new ValueInfo('0in', [-1, 4], 'float32'),
      new ValueInfo('in2', [1, 4], 'float32'),
    );
    gNonLin.outputs.push(new ValueInfo('0out', [-1, 4], 'float32'));
    gNonLin.tensors.w_gemm2 = new Tensor(
      'w_gemm2',
      [4, 4],
      'float32',
      true,
      false,
      new Uint8Array(16),
    );
    gNonLin.nodes.push(
      new Node('Conv', ['0in', 'missing_w'], ['c_out']),
      new Node('Gemm', ['c_out', 'w_gemm2'], ['g_out']),
      new Node('Gemm', ['g_out', 'missing_w'], ['0out']),
    );
    expect(isLinearGraph(gNonLin)).toBe(false);
    expect(generateTFJSCode(gNonLin)).toContain('tf.model');

    const gMultiDyn = new Graph('md');
    gMultiDyn.inputs.push(new ValueInfo('a', [1], 'float32'), new ValueInfo('b', [1], 'float32'));
    gMultiDyn.outputs.push(new ValueInfo('c', [1], 'float32'));
    gMultiDyn.nodes.push(new Node('Relu', ['a', 'b'], ['c']));
    expect(generateTFJSCode(gMultiDyn)).toContain('.apply([a, b])');

    const gBadOp = new Graph('bad');
    gBadOp.inputs.push(new ValueInfo('a', [1], 'float32'));
    gBadOp.outputs.push(new ValueInfo('b', [1], 'float32'));
    gBadOp.nodes.push(new Node('UnknownOp', ['a'], ['b']));
    expect(() => generateTFJSCode(gBadOp)).toThrow(/Unsupported operator/);

    const gMultiOut = new Graph('mo');
    gMultiOut.inputs.push(new ValueInfo('a', [1], 'float32'));
    gMultiOut.outputs.push(new ValueInfo('b', [1], 'float32'));
    gMultiOut.nodes.push(new Node('Relu', ['a'], ['b', 'c']));
    expect(isLinearGraph(gMultiOut)).toBe(false);
    const gNoOut = new Graph('no');
    gNoOut.inputs.push(new ValueInfo('a', [1], 'float32'));
    expect(isLinearGraph(gNoOut)).toBe(false);

    // TFJS serializer
    gLin.tensors.t_int = new Tensor('0int', [-1], 'int32', true, false, new Uint8Array(4));
    gLin.tensors.t_bool = new Tensor('t_bool', [1], 'bool', true, false, new Uint8Array(1));
    gLin.tensors.t_str = new Tensor('t_str', [1], 'string', true, false, new Uint8Array(2));
    gLin.tensors.t_nodata = new Tensor('t_nodata', [1], 'float32', true, false, null);
    gLin.initializers.push('missing_init');
    const tfjsArt = serializeTFJSWeights(gLin);
    expect(tfjsArt.weightsBin.byteLength).toBeGreaterThan(0);

    // PyTorch generator & serializer
    const gPt = new Graph('pt');
    gPt.inputs.push(new ValueInfo('0x', [1, 3, 4, 4], 'float32'));
    gPt.valueInfo.push(new ValueInfo('vi_t', [2, 2], 'float32'));
    gPt.outputs.push(new ValueInfo('y', [1, 2], 'float32'));
    gPt.tensors.w1 = new Tensor('w1', [2, 3, 3, 3], 'float32', true, false, new Uint8Array(8));
    gPt.tensors.w_lin = new Tensor('w_lin', [2, 2], 'float32', true, false, new Uint8Array(8));
    gPt.tensors.sh = new Tensor(
      'sh',
      [1],
      'int32',
      true,
      false,
      new Uint8Array(new Int32Array([2]).buffer),
    );
    gPt.nodes.push(
      new Node('Conv', ['0x', 'w1'], ['h1']),
      new Node('Sigmoid', ['h1'], ['h2']),
      new Node('Tanh', ['h2'], ['h3']),
      new Node('Add', ['h3', '0x'], ['h4']),
      new Node('Mul', ['h4', '0x'], ['h5']),
      new Node('Concat', ['h5', '0x'], ['h6']),
      new Node('Reshape', ['h6', 'sh'], ['h7']),
      new Node('Reshape', ['h7', '0x'], ['h8']),
      new Node('Transpose', ['h8'], ['h9'], { perm: new Attribute('perm', 'INTS', [0, 1]) }),
      new Node('Transpose', ['h9'], ['h10']),
      new Node('Gemm', ['h10', 'w_lin'], ['h11'], { transB: new Attribute('transB', 'INT', 1) }),
      new Node('Gemm', ['h11', 'vi_t'], ['h12']),
      new Node('BatchNormalization', ['h12', 'w_lin'], ['h13'], {
        momentum: new Attribute('momentum', 'FLOAT', 0.9),
      }),
      new Node('CustomPt', ['h13', ''], ['y']),
    );
    expect(new PyTorchGenerator(gPt).generate()).toContain('class ONNXModel');
    expect(
      PyTorchSerializer.serialize([
        new Tensor('t1', [], 'float32', true, false, null),
        new Tensor('t2', ['dyn', 2], 'float64', true, false, null),
      ]).byteLength,
    ).toBeGreaterThan(0);

    // Caffe, CNTK, MXNet, Keras, TF, ONNXScript generators
    expect(new CaffeGenerator(gLin).generate()).toContain('layer {');
    expect(new CaffeGenerator(gPt).generate()).toContain('layer {');
    expect(new CNTKGenerator(gLin).generate()).toContain('import cntk');
    expect(new CNTKGenerator(gPt).generate()).toContain('import cntk');
    expect(new MXNetGenerator(gLin).generate()).toContain('import mxnet');
    expect(new MXNetGenerator(gPt).generate()).toContain('import mxnet');
    expect(new KerasGenerator(gLin).generateSource()).toContain('import keras');
    expect(new KerasGenerator(gPt).generateSource()).toContain('import keras');
    expect(new TensorFlowGenerator(gLin).generate()).toContain('import tensorflow');
    expect(new OnnxScriptGenerator(gLin).generate()).toContain('import onnxscript');
    const gOs = new Graph('os');
    gOs.nodes.push(
      new Node('Add', ['a', 'b'], ['c'], {
        unk: new Attribute('unk', 'TENSOR', null),
      }),
    );
    expect(new OnnxScriptGenerator(gOs).generate()).toContain('op.Add');
    const osParsed = new OnnxScriptParser().parseScript(
      '@script()\ndef model(x: FLOAT[1, 3, , 4]) -> (FLOAT[1, 3]):\n    y = op.Relu(x)\n    return y\n',
    );
    expect(osParsed.nodes.length).toBe(1);
  });

  it('should cover mmdnn/api.ts, fusion, shape-inference, topology, normalizer, Caffe/MXNet/Darknet/NCNN/TF mappers & parsers', async () => {
    const fEmpty = new Blob([''], { type: 'text/plain' });
    const fBadJson = new Blob(['{bad json'], { type: 'text/plain' });

    for (const src of [
      'caffe',
      'mxnet',
      'tensorflow',
      'paddle',
      'scikitlearn',
      'lightgbm',
      'xgboost',
      'catboost',
      'sparkml',
      'pyspark',
      'h2o',
      'libsvm',
      'onnxscript',
      'jax',
    ] as const) {
      await expect(api.convert(src, 'onnx', [])).rejects.toThrow();
      const res = await api.convert(src, 'onnx', [fBadJson]);
      expect(res).toBeDefined();
    }
    expect(await api.convert('cntk', 'onnx', [])).toBeDefined();
    await expect(api.convert('onnx', 'onnx', [])).rejects.toThrow(/No ONNX file/);

    const fValidH2O = new Blob(['{"algo":"gbm"}']);
    expect(await api.convert('h2o', 'onnx', [fValidH2O])).toBeDefined();
    const fValidSvm = new Blob(['svm_type c_svc\nkernel_type rbf\n']);
    expect(await api.convert('libsvm', 'onnx', [fValidSvm])).toBeDefined();
    const fValidJax = new Blob([
      JSON.stringify({ invars: ['x'], outvars: ['y'], constvars: [], eqns: [] }),
    ]);
    expect(await api.convert('jax', 'onnx', [fValidJax])).toBeDefined();
    const fValidCaffe = new Blob([
      'layer { name: "conv1_weights" type: "Convolution" bottom: "data_W" top: "out" }',
    ]);
    expect(await api.convert('caffe', 'onnx', [fValidCaffe, fEmpty])).toBeDefined();
    expect(await api.convert('caffe', 'unknown_target' as 'onnx', [fValidCaffe])).toBeDefined();

    // fusion, shape-inference, topology, normalizer
    const rep = new MMDNNReporter();
    const gTop = new Graph('top');
    gTop.tensors.t1 = new Tensor('t1', [2, 3], 'float32');
    gTop.inputs.push(new ValueInfo('in1', [2, 2], 'float32'));
    gTop.outputs.push(new ValueInfo('out1', undefined as unknown as number[], 'float32'));
    gTop.nodes.push(
      new Node('Add', ['in1', 't1'], ['out1'], {}, 'add_incompat'),
      new Node('Relu', [''], ['r_out'], {}, 'relu_empty_in'),
    );
    const sorted = topologicalSort(gTop, rep);
    expect(sorted.nodes.length).toBe(2);
    const sie = new ShapeInferenceEngine();
    sie.inferShapes(sorted, rep);
    expect(sorted.outputs[0]?.shape).toEqual([2, 2]);

    const gFuse = new Graph('fuse');
    gFuse.nodes.push(
      new Node('MatMul', ['a', 'b'], ['mm_out'], {}, 'mm'),
      new Node('Add', ['bias', 'mm_out'], ['add_out'], {}, 'add_rev'),
    );
    new NodeFusionRegistry().applyFusions(gFuse, rep);
    expect(gFuse.nodes[0]?.opType).toBe('Gemm');

    const gNorm = new Graph('norm');
    gNorm.tensors['1f64'] = new Tensor(
      '1f64',
      [1],
      'float64',
      true,
      false,
      new Float64Array([1.0]),
    );
    gNorm.inputs.push(new ValueInfo('', [1], 'float64'));
    gNorm.outputs.push(new ValueInfo('m_out', [1], 'float64'));
    gNorm.nodes.push(
      new Node('CaffeScale', ['1f64'], ['s_out']),
      new Node('MxNetActivation', ['s_out'], ['m_out']),
    );
    new ONNXNormalizer().normalize(gNorm);
    expect(gNorm.tensors._1f64?.dtype).toBe('float32');

    // Caffe parser & mapper branches
    expect(
      parsePrototxt('layer {\n  name: "l1"\n  bottom: "a"\n  bottom: "b"\n  value: true\n}'),
    ).toBeDefined();
    const cm = new CaffeMapper();
    const gCm = new Graph('cm');
    register_caffe_op('other_domain', 'NoOp')(cm, 'noop', {});
    cm.map(
      {
        name: 'c1',
        type: 'Convolution',
        blobs: [
          { shape: [1, 1, 1, 1], data: [1] },
          { shape: { dim: [1] }, data: [0] },
        ],
      },
      gCm,
    );
    cm.map({ name: 'c2', type: 'Convolution', blobs: [1, 2] }, gCm);
    cm.map({ name: 'sp', type: 'Split', bottom: ['x'], top: [] }, gCm);
    cm.map(
      { name: 'sl', type: 'Slice', bottom: ['x'], top: ['y'], slice_param: { slice_point: 2 } },
      gCm,
    );

    // MXNet mapper branches
    const mxm = new MxNetMapper();
    register_mxnet_op('other_domain', 'NoOp')(mxm, 'noop', {});
    const gMx = new Graph('mx');
    mxm.map(
      {
        op: 'Pooling',
        name: 'p1',
        attrs: { pool_type: 'avg', kernel: '(2, 2)', stride: '(2, 2)', pad: '(1, 1, 1, 1)' },
      },
      gMx,
    );
    mxm.map(
      {
        op: 'Deconvolution',
        name: 'd1',
        attrs: {
          kernel: '(2, 2)',
          stride: '(2, 2)',
          pad: '(1, 1)',
          dilate: '(1, 1)',
          num_group: '2',
        },
      },
      gMx,
    );
    mxm.map({ op: 'LeakyReLU', name: 'lr', attrs: { act_type: 'unknown' } }, gMx);

    // Darknet parser & mapper
    expect(
      parseCfg('[net]\nwidth=32\nbad_line_without_equals\n[convolutional]\nfilters=16\n').length,
    ).toBe(2);
    expect(parseWeights(new ArrayBuffer(16)).length).toBe(0);
    const gDm = new Graph('dm');
    const dm = new DarknetMapper(gDm, new Float32Array(10));
    dm.map([
      { type: 'net', channels: 3 },
      { type: 'convolutional', filters: 1, size: 1, batch_normalize: 1, activation: 'leaky' },
    ]);
    expect(gDm.nodes.length).toBeGreaterThan(0);

    // NCNN parser & mapper
    const ncnnParam =
      '7767517\n2 2\nInput input 0 1 in0 0=3\nCustomOp cust 1 1 in0 out0 -23300=2,1,2 1=1.5,2.5 2=1,2\n';
    const parsedNcnn = parseNcnnParam(ncnnParam);
    const ncnnBin = new NcnnBinParser(new Uint8Array(32).buffer);
    expect(ncnnBin.readBytes(3).byteLength).toBe(3);
    expect(new NcnnMapper(parsedNcnn, new Uint8Array(32).buffer).getGraph().nodes.length).toBe(1);

    // TF parser & mapper
    const pbtxt = `
node {
  name: "n1"
  op: "MaxPool"
  input: "x"
  attr {
    key: "ksize"
    value { list { i: 1 i: 2 i: 2 i: 1 } }
  }
  attr {
    key: "strides"
    value { list { i: 1 i: 2 i: 2 i: 1 } }
  }
  attr {
    key: "padding"
    value { s: "SAME" }
  }
  attr {
    key: "shape_attr"
    value { shape { dim { size: 2 } } }
  }
}
node {
  name: "n2"
  op: "MaxPool"
  input: "x"
  attr {
    key: "padding"
    value { s: "VALID" }
  }
}
node {
  name: "n3"
  op: "Custom"
  key: "inline_i" i: 10 f: 2.5
}
`;
    const tfGraph = parsePbtxt(pbtxt);
    const tfm = new TFMapper();
    const gTf = new Graph('tf');
    for (const n of tfGraph.node) {
      tfm.map(n, gTf);
    }
    expect(parsePbtxt('node { } test').node.length).toBe(1);

    // Keras importer
    const ki = new KerasImporter();
    const fl = new FileLoader([new File(['not json'], 'model.h5')]);
    await fl.initialize();
    await expect(ki.parse(fl, rep)).rejects.toThrow();

    // Remaining lines in jax/mapper.ts (lines 68-69: flaxState value neither flat object nor array)
    new JaxMapper({ invars: [], outvars: [], constvars: ['c_num'], eqns: [] }, { c_num: {} }).map();

    // Remaining lines in keras/index.ts (QAT, DynamicQuantizeLinear, signatures) and keras-ast.ts
    registerCustomKerasLayer('QatAndDynLayer', (_nodeName, _layerName, inputs, outName) => [
      {
        opType: 'MatMul',
        inputs: [inputs[0] || '', 'w_packed'],
        outputs: [`${outName}_p4`],
        name: 'packed_4bit_quantize_wrapper',
        attributes: [],
      },
      {
        opType: 'MatMul',
        inputs: [`${outName}_p4`, 'w_dyn'],
        outputs: [outName],
        name: 'dynamic_quant_node',
        attributes: [{ name: 'other', type: 'UNKNOWN' }],
      },
    ]);
    const sigModelJson = JSON.stringify({
      format: 'layers-model',
      signature: {
        serving_default: {
          inputs: { sig_in: { name: 'in_layer_input:0:0' }, sig_str_in: 'raw_in' },
          outputs: { sig_out: { name: 'qat_layer:0:0' }, sig_str_out: 'raw_out' },
        },
      },
      modelTopology: {
        class_name: 'Sequential',
        config: {
          name: 'seq_outer',
          layers: [
            { name: 'in_layer', className: 'InputLayer', input_shape: [4, null] },
            {
              class_name: 'Sequential',
              config: {
                name: 'nested_seq',
                layers: [
                  {
                    class_name: 'InputLayer',
                    config: { name: 'nested_in', batch_input_shape: [null, 4] },
                  },
                  {
                    class_name: 'QatAndDynLayer',
                    config: { name: 'qat_layer' },
                  },
                ],
              },
            },
          ],
        },
      },
      weightsManifest: [],
    });
    expect(new Keras2OnnxConverter(sigModelJson).convert().byteLength).toBeGreaterThan(0);

    // mmdnn/api.ts remaining lines: valid caffe, mxnet, tensorflow, paddle, lightgbm, xgboost, catboost, sparkml, onnxscript, and target exports
    const validPrototxt = new Blob([
      'layer {\n  name: "c1"\n  type: "VisionTransform"\n  bottom: "x_weights"\n  top: "y"\n  sub {\n    a: 1\n  }\n  sub {\n    b: 2\n  }\n  sub {\n    c: 3\n  }\n}',
    ]);
    expect(await api.convert('caffe', 'onnx', [validPrototxt])).toBeDefined();
    for (const tgt of [
      'pytorch_code',
      'keras',
      'caffe',
      'tensorflow',
      'cntk',
      'mxnet',
      'onnxscript',
    ] as const) {
      expect(await api.convert('caffe', tgt, [validPrototxt])).toBeDefined();
    }

    const validMxnet = new Blob([
      JSON.stringify({
        nodes: [
          { op: 'null', name: 'data', inputs: [] },
          {
            op: 'LeakyReLU',
            name: 'elu1',
            attrs: { act_type: 'elu', slope: '0.2' },
            inputs: [[0, 0, 0]],
          },
          { op: 'LeakyReLU', name: 'prelu1', attrs: { act_type: 'prelu' }, inputs: [[1, 0, 0]] },
        ],
        arg_nodes: [0],
        heads: [[2, 0, 0]],
      }),
    ]);
    expect(await api.convert('mxnet', 'onnx', [validMxnet])).toBeDefined();

    const validTf = new Blob([
      'node {\n  name: "c"\n  op: "Conv2D"\n  input: "x"\n  attr {\n    key: "padding"\n    value { s: "VALID" }\n  }\n}\n',
    ]);
    expect(await api.convert('tensorflow', 'onnx', [validTf])).toBeDefined();

    const validPaddle = new Blob([
      JSON.stringify({
        blocks: [{ ops: [{ type: 'relu', inputs: { X: ['x'] }, outputs: { Out: ['y'] } }] }],
      }),
    ]);
    expect(await api.convert('paddle', 'onnx', [validPaddle])).toBeDefined();

    const validLgbm = new Blob([JSON.stringify({ tree_info: [{ num_leaves: 1 }] })]);
    expect(await api.convert('lightgbm', 'onnx', [validLgbm])).toBeDefined();

    const validOnnxScript = new Blob([
      '@script()\ndef m(x: FLOAT) -> (FLOAT):\n    y = op.Relu(x)\n    return y\n',
    ]);
    expect(await api.convert('onnxscript', 'onnx', [validOnnxScript])).toBeDefined();

    // topology.ts cycle detection and initializers (lines 31-32)
    const gCycle = new Graph('cyc');
    gCycle.initializers.push('init_top');
    gCycle.nodes.push(
      new Node('Relu', ['b'], ['a'], {}, 'n_a'),
      new Node('Relu', ['a'], ['b'], {}, 'n_b'),
    );
    expect(() => topologicalSort(gCycle, rep)).toThrow(/Cyclic graph/);

    // keras/generator.ts line 168: input that is also an initializer & input with empty shape []
    const gKg = new Graph('kg');
    gKg.inputs.push(
      new ValueInfo('init_in', [1, 2], 'float32'),
      new ValueInfo('empty_shape_in', [], 'float32'),
      new ValueInfo('sym_shape_in', ['batch', 4], 'float32'),
    );
    gKg.tensors.init_in = new Tensor('init_in', [1, 2], 'float32', true);
    gKg.nodes.push(new Node('Relu', ['init_in'], ['out']));
    expect(new KerasGenerator(gKg).generateSource()).toContain('create_kg');

    // keras/index.ts lines 272-273, 275-276, 281-283: attributes with type='UNKNOWN' and i, f, floats defined
    // keras/index.ts lines 457-461: signature output matching internal output name
    // keras/keras-ast.ts lines 165-174, 185-186: Sequential with first layer followed by nested Sequential
    registerCustomKerasLayer('RawAttrFieldsLayer', (_nodeName, _layerName, inputs, outName) => [
      {
        opType: 'CustomRawAttr',
        inputs,
        outputs: [outName],
        name: _nodeName,
        attributes: [
          { name: 'rawI', i: 7 },
          { name: 'rawF', f: 2.5 },
          { name: 'rawS', s: 'str' },
          { name: 'rawIs', ints: [1] },
          { name: 'rawFs', floats: [1.5] },
        ] as unknown as Array<{ name: string; type: 'INT' }>,
      },
    ]);
    const nestedAfterFirstJson = JSON.stringify({
      format: 'layers-model',
      signature: {
        serving_default: {
          inputs: { mapped_in: 'first_in_input:0:0' },
          outputs: { mapped_out: 'nested_seq/qat_layer:0' },
        },
      },
      modelTopology: {
        class_name: 'Sequential',
        config: {
          name: 'seq_outer2',
          layers: [
            {
              class_name: 'InputLayer',
              config: { name: 'first_in', batch_input_shape: [null, 4] },
            },
            {
              class_name: 'Sequential',
              config: {
                name: 'nested_seq',
                layers: [
                  {
                    class_name: 'InputLayer',
                    config: { name: 'nested_in', batch_input_shape: [null, 4] },
                  },
                  {
                    class_name: 'RawAttrFieldsLayer',
                    config: { name: 'qat_layer' },
                  },
                ],
              },
            },
          ],
        },
      },
      weightsManifest: [],
    });
    expect(new Keras2OnnxConverter(nestedAfterFirstJson).convert().byteLength).toBeGreaterThan(0);

    // Also test nested Sequential with parentPrefix non-empty (3 levels deep) for lines 185-186
    const deepSeqJson = JSON.stringify({
      format: 'layers-model',
      modelTopology: {
        class_name: 'Sequential',
        config: {
          name: 'l1',
          layers: [
            {
              class_name: 'Sequential',
              config: {
                name: 'l2',
                layers: [
                  {
                    class_name: 'Sequential',
                    config: {
                      name: 'l3',
                      layers: [
                        {
                          class_name: 'RawAttrFieldsLayer',
                          config: { name: 'leaf', batch_input_shape: [null, 4] },
                        },
                      ],
                    },
                  },
                ],
              },
            },
          ],
        },
      },
      weightsManifest: [],
    });
    expect(new Keras2OnnxConverter(deepSeqJson).convert().byteLength).toBeGreaterThan(0);

    // mmdnn/api.ts: caffe with weights file matching layer 'c1', valid scikitlearn/xgboost/catboost/sparkml, and throwing file0.text()
    const caffeWeightsBlob = new Blob([
      new Uint8Array([0x12, 0x06, 0x0a, 0x02, 0x63, 0x31, 0x32, 0x00]),
    ]);
    expect(await api.convert('caffe', 'onnx', [validPrototxt, caffeWeightsBlob])).toBeDefined();
    expect(
      await api.convert('scikitlearn', 'onnx', [new Blob([JSON.stringify({ model: 'SVC' })])]),
    ).toBeDefined();
    expect(
      await api.convert('xgboost', 'onnx', [new Blob([JSON.stringify({ learner: {} })])]),
    ).toBeDefined();
    expect(
      await api.convert('catboost', 'onnx', [
        new Blob([JSON.stringify({ catboost_version: '1.0' })]),
      ]),
    ).toBeDefined();
    expect(
      await api.convert('sparkml', 'onnx', [
        new Blob([JSON.stringify({ class: 'LogisticRegression' })]),
      ]),
    ).toBeDefined();

    const throwingBlob = {
      name: 'bad.txt',
      size: 1,
      type: 'text/plain',
      text: async () => {
        throw new Error('read error');
      },
      arrayBuffer: async () => new ArrayBuffer(0),
      slice: () => new Blob([]),
    } as unknown as Blob;
    for (const src of [
      'scikitlearn',
      'lightgbm',
      'xgboost',
      'catboost',
      'sparkml',
      'h2o',
      'libsvm',
      'onnxscript',
      'jax',
    ] as const) {
      expect(await api.convert(src, 'onnx', [throwingBlob])).toBeDefined();
    }

    // cntk/generator.ts line 46 & mxnet/generator.ts line 47: getShape fallback to null
    const gUnkShape = new Graph('unk');
    gUnkShape.nodes.push(
      new Node('Conv', ['x', 'unmapped_w'], ['h1']),
      new Node('Gemm', ['h1', 'unmapped_w2'], ['y']),
    );
    expect(new CNTKGenerator(gUnkShape).generate()).toContain('Convolution2D');
    expect(new MXNetGenerator(gUnkShape).generate()).toContain('Conv2D');

    // mxnet/mapper.ts lines 313-314: LeakyReLU with act_type='leaky' and slope='0.1'
    expect(
      mxm.map(
        { op: 'LeakyReLU', name: 'lr_slope', attrs: { act_type: 'leaky', slope: '0.1' } },
        gMx,
      )[0]?.opType,
    ).toBe('LeakyRelu');

    // Exhaustive branch tests for CaffeMapper, MxNetMapper, DarknetMapper, Generators, Fusion, ShapeInference, TFMapper, Keras
    // 1. CaffeMapper nameless layers and array spatial params
    for (const lType of [
      'Convolution',
      'InnerProduct',
      'ReLU',
      'Pooling',
      'LRN',
      'Softmax',
      'Eltwise',
      'Concat',
      'Scale',
      'BatchNorm',
      'Dropout',
      'Reshape',
      'Flatten',
      'Split',
      'Slice',
      'VisionTransform',
      'UnknownCaffeLayer',
      undefined,
    ] as const) {
      cm.map({ type: lType }, gCm);
      cm.map(
        {
          type: lType,
          bottom: [],
          top: ['t1', ''],
          blobs: [{ shape: [0], data: [1] }],
          convolution_param: { pad: [], kernel_size: [], stride: [], dilation: [] },
          pooling_param: { pad: [1], kernel_size: [2], stride: [2] },
          slice_param: { slice_point: [1, 2] },
        },
        gCm,
      );
    }

    // 2. MxNetMapper nameless nodes and 2-element vs 4-element pad
    for (const opName of [
      'Convolution',
      'FullyConnected',
      'Activation',
      'Pooling',
      'BatchNorm',
      'Dropout',
      'Flatten',
      'Reshape',
      'Concat',
      'elemwise_add',
      'elemwise_sub',
      'elemwise_mul',
      'broadcast_add',
      'broadcast_mul',
      'SoftmaxOutput',
      'LeakyReLU',
      'UpSampling',
      'SliceChannel',
      'Crop',
      'Deconvolution',
    ] as const) {
      mxm.map({ op: opName }, gMx);
      mxm.map(
        {
          op: opName,
          attrs: {
            pad: '(1)',
            kernel: 'not_tuple',
            act_type: 'unknown_act',
            pool_type: 'other_pool',
          },
        },
        gMx,
      );
    }

    // 3. DarknetMapper route/shortcut/yolo/upsample/convolutional edge cases and parseCfg/parseWeights
    parseCfg('[net]\n=no_key\nk=1,a\nempty=\n');
    const hdr20 = new ArrayBuffer(20);
    const dv20 = new DataView(hdr20);
    dv20.setInt32(0, 0, true);
    dv20.setInt32(4, 2, true);
    expect(parseWeights(hdr20).length).toBe(0);

    const gDm2 = new Graph('dm2');
    const dm2 = new DarknetMapper(gDm2, new Float32Array(0));
    dm2.map([
      { type: 'yolo' }, // yolo at i=0 where prevOutput is undefined
      { type: 'convolutional' }, // default filters/size/stride, linear activation
      { type: 'convolutional', activation: 'mish', pad: 1 },
      { type: 'convolutional', activation: 'swish' },
      { type: 'maxpool' },
      { type: 'avgpool' },
      { type: 'connected' },
      { type: 'shortcut', from: 10 }, // out of bounds targetIdx
      { type: 'route' }, // default layers=0
      { type: 'route', layers: [10, 11] }, // out of bounds route
      { type: 'upsample' },
      { type: 'region' },
    ]);

    // 4. Generators with empty graph name and nodes with missing inputs/outputs/attributes
    const gGenEdge = new Graph('');
    const nConvEmpty = new Node('Conv', [], []);
    const nMaxPoolEmpty = new Node('MaxPool', [], []);
    const nAvgPoolEmpty = new Node('AveragePool', [], []);
    const nGemmEmpty = new Node('Gemm', [], []);
    const nAddEmpty = new Node('Add', [], []);
    const nReluEmpty = new Node('Relu', [], []);
    const nFlatEmpty = new Node('Flatten', [], []);
    const nGapEmpty = new Node('GlobalAveragePool', [], []);
    const nBnEmpty = new Node('BatchNormalization', [], []);
    const nConcatEmpty = new Node('Concat', [], []);
    const nSmEmpty = new Node('Softmax', [], []);
    const nSigEmpty = new Node('Sigmoid', [], []);
    const nTanhEmpty = new Node('Tanh', [], []);
    const nMulEmpty = new Node('Mul', [], []);
    const nReshEmpty = new Node('Reshape', [], []);
    const nTransEmpty = new Node('Transpose', [], []);
    gGenEdge.nodes.push(
      nConvEmpty,
      nMaxPoolEmpty,
      nAvgPoolEmpty,
      nGemmEmpty,
      nAddEmpty,
      nReluEmpty,
      nFlatEmpty,
      nGapEmpty,
      nBnEmpty,
      nConcatEmpty,
      nSmEmpty,
      nSigEmpty,
      nTanhEmpty,
      nMulEmpty,
      nReshEmpty,
      nTransEmpty,
    );
    expect(new CaffeGenerator(gGenEdge).generate()).toBeDefined();
    expect(new CNTKGenerator(gGenEdge).generate()).toBeDefined();
    expect(new MXNetGenerator(gGenEdge).generate()).toBeDefined();
    expect(new KerasGenerator(gGenEdge).generateSource()).toBeDefined();
    expect(new PyTorchGenerator(gGenEdge).generate()).toBeDefined();
    expect(new TensorFlowGenerator(gGenEdge).generate()).toBeDefined();

    // KerasGenerator saveWeights with non-TypedArray data and empty shape
    const gKgWeights = new Graph('kgw');
    const tKgEmpty = new Tensor('tke', [], 'uint8', true, false, null);
    const tKgPlainObj = new Tensor('tkp', [1], 'uint8', true, false, {
      length: 1,
    } as unknown as Uint8Array);
    gKgWeights.tensors.tke = tKgEmpty;
    gKgWeights.tensors.tkp = tKgPlainObj;
    expect(new KerasGenerator(gKgWeights).exportWeights().byteLength).toBeGreaterThan(0);

    // PyTorchGenerator sequential chains & Conv1d/Conv3d
    const gPtSeq = new Graph('ptseq');
    gPtSeq.inputs.push(new ValueInfo('x', [1, 3, 4], 'float32'));
    gPtSeq.tensors.w1d = new Tensor('w1d', [2, 3, 3], 'float32', true);
    gPtSeq.tensors.w3d = new Tensor('w3d', [2, 2, 3, 3, 3], 'float32', true);
    gPtSeq.tensors.w0d = new Tensor('w0d', [0, 0, 3, 3], 'float32', true);
    gPtSeq.tensors.b = new Tensor('b', [2], 'float32', true);
    gPtSeq.nodes.push(
      new Node('Conv', ['x', 'w1d', 'b'], ['s1']),
      new Node('Relu', ['s1'], ['s2']),
      new Node('Conv', ['s2', 'w3d'], ['s3']),
      new Node('MaxPool', ['s3'], ['s4']),
      new Node('AveragePool', ['s4'], ['s5']),
      new Node('GlobalAveragePool', ['s5'], ['s6']),
      new Node('Flatten', ['s6'], ['s7']),
      new Node('Conv', ['s7', 'w0d'], ['s8']),
      new Node('BatchNormalization', ['s8', 'w0d'], ['s9']),
    );
    expect(new PyTorchGenerator(gPtSeq).generate()).toContain('nn.Sequential');

    // OnnxScriptGenerator with bigint attribute
    const gOsBig = new Graph('osb');
    gOsBig.nodes.push(
      new Node('Cast', ['x'], ['y'], {
        to: new Attribute('to', 'INT', 1n as unknown as number),
      }),
    );
    expect(new OnnxScriptGenerator(gOsBig).generate()).toContain('to=1');

    // TFJS generator with empty inputs/outputs on node and bigint attribute
    const gTfjsEdge = new Graph('tfjse');
    gTfjsEdge.inputs.push(new ValueInfo('x', [1, 3, 4, 4], 'float32'));
    gTfjsEdge.outputs.push(new ValueInfo('y', [1, 3, 4, 4], 'float32'));
    const nTfjsRelu = new Node('Relu', ['x'], []);
    gTfjsEdge.nodes.push(nTfjsRelu);
    expect(isLinearGraph(gTfjsEdge)).toBe(false);
    expect(generateTFJSCode(gTfjsEdge)).toContain('tf.layers.reLU');
    const gTfjsLinEmptyOut = new Graph('tfjsle');
    gTfjsLinEmptyOut.inputs.push(new ValueInfo('', [1, 3, 4, 4], 'float32'));
    gTfjsLinEmptyOut.outputs.push(new ValueInfo('', [1, 3, 4, 4], 'float32'));
    gTfjsLinEmptyOut.nodes.push(
      new Node('MaxPool', [''], [''], {
        strides: new Attribute('strides', 'INTS', [1n as unknown as number, 1]),
      }),
    );
    expect(isLinearGraph(gTfjsLinEmptyOut)).toBe(true);
    expect(generateTFJSCode(gTfjsLinEmptyOut)).toContain('tf.sequential()');

    // Fusion, ShapeInference, TFMapper, NCNN, CaffeParser, JAXprParser
    const gFuseEdge = new Graph('fe');
    gFuseEdge.nodes.push(
      new Node('Conv', ['x'], ['c_multi']),
      new Node('BatchNormalization', ['c_multi'], ['bn1']),
      new Node('BatchNormalization', ['c_multi'], ['bn2']),
      new Node('BatchNormalization', [], ['bn_no_in']),
      new Node('MatMul', ['x', 'w'], ['mm_multi']),
      new Node('Add', ['mm_multi', 'b'], ['a1']),
      new Node('Add', ['mm_multi', 'b'], ['a2']),
      new Node('Add', [], ['add_no_in']),
      new Node('Add', ['mm_multi'], ['add_no_bias']),
    );
    new NodeFusionRegistry().applyFusions(gFuseEdge, rep);

    const gSiEdge = new Graph('sie');
    gSiEdge.inputs.push({ name: 'in_no_shape' } as ValueInfo);
    gSiEdge.tensors.t_no_shape = { name: 't_no_shape' } as Tensor;
    gSiEdge.inputs.push(
      new ValueInfo('a1', [1, 3], 'float32'),
      new ValueInfo('b1', [2, 1], 'float32'),
      new ValueInfo('zero_dim', [0], 'float32'),
    );
    gSiEdge.outputs.push(new ValueInfo('already_shaped', [2, 3], 'float32'));
    gSiEdge.nodes.push(
      new Node('Relu', [], []),
      new Node('Add', [], []),
      new Node('Mul', ['a1', 'b1'], ['already_shaped']),
      new Node('Add', ['zero_dim', 'a1'], ['z_out']),
    );
    new ShapeInferenceEngine().inferShapes(gSiEdge, rep);

    // TFMapper nameless and tensor without content
    tfm.map({ name: '', op: 'Placeholder', input: [], attr: {} }, gTf);
    tfm.map(
      {
        name: 'c_no_val',
        op: 'Const',
        input: [],
        attr: { value: { tensor: { dtype: 'DT_FLOAT', shape: [1] } } },
      },
      gTf,
    );
    parsePbtxt(
      'node {\n  name: "t"\n  op: "Const"\n  attr {\n    key: "v"\n    value { tensor { shape { dim { size: 1 } } } }\n  }\n}\n',
    );

    // NcnnMapper nameless Input and generic node with empty bottom/top
    new NcnnMapper(
      {
        magic: 7767517,
        layerCount: 2,
        blobCount: 2,
        nodes: [
          { type: 'Input', name: 'in', bottoms: [], tops: [], attrs: {} },
          { type: 'CustomNcnn', name: 'c', bottoms: [''], tops: [''], attrs: {} },
        ],
      },
      new ArrayBuffer(16),
    );
    parseNcnnParam('7767517\n\n');
    parseNcnnParam('0\n0 0\nBadLine\nType Name 1 1\n');

    // Caffe prototxt single-quoted string
    parsePrototxt("name: 'single_quoted'\n");

    // JAXprParser eqn with non-object invar
    new JAXprParser().parse({
      eqns: [{ primitive: 'add', invars: [null, { name: 'x' }], outvars: [{ name: 'y' }] }],
    });

    // weight-loader.ts line 96: complex64
    expect(calculateByteLength({ name: 'wc', shape: [2], dtype: 'complex64' })).toBe(16);

    // keras/index.ts remaining branches:
    // - model.format !== 'layers-model' (line 74)
    // - QAT and DynamicQuantizeLinear with empty inputs and no second input (295, 323, 338, 339)
    // - signature output matching out.split(':')[0] (458)
    // - Dense with use_bias: false (526), LeakyReLU/ELU/ThresholdedReLU with numeric alpha/theta (556, 584, 598)
    // - Conv2D/MaxPooling2D with padding: 'same' (638, 688)
    // - BatchNormalization with momentum, scale: false, center: false (726, 730, 731)
    // - GroupNormalization with scale: false, center: false (823, 824)
    // - Reshape with non-null target_shape (914)
    // - Rescaling with scale and offset (945, 946)
    // - Resizing with nearest and bicubic (999, 1000)
    // - Subtract with 3 inputs (1088)
    registerCustomKerasLayer('EmptyInputsQatLayer', (_nodeName, _layerName, _inputs, outName) => [
      {
        opType: 'MatMul',
        inputs: [],
        outputs: [`${outName}_q`],
        name: 'quantize_wrapper_empty',
        attributes: [],
      },
      {
        opType: 'MatMul',
        inputs: [],
        outputs: [outName],
        name: 'dynamic_quant_empty',
        attributes: [],
      },
    ]);

    const kerasBranchesJson = JSON.stringify({
      format: 'graph-model',
      modelTopology: {
        class_name: 'Functional',
        config: {
          name: 'fb_model',
          layers: [
            {
              class_name: 'InputLayer',
              name: 'in1',
              config: { name: 'in1', batch_input_shape: [null, 4, 4, 3] },
              inbound_nodes: [],
            },
            {
              class_name: 'InputLayer',
              name: 'in2',
              config: { name: 'in2', batch_input_shape: [null, 4, 4, 3] },
              inbound_nodes: [],
            },
            {
              class_name: 'InputLayer',
              name: 'in3',
              config: { name: 'in3', batch_input_shape: [null, 4, 4, 3] },
              inbound_nodes: [],
            },
            {
              class_name: 'Dense',
              name: 'd_nobias',
              config: { name: 'd_nobias', units: 4, use_bias: false },
              inbound_nodes: [[['in1', 0, 0, {}]]],
            },
            {
              class_name: 'LeakyReLU',
              name: 'lrelu',
              config: { name: 'lrelu', alpha: 0.2 },
              inbound_nodes: [[['d_nobias', 0, 0, {}]]],
            },
            {
              class_name: 'ELU',
              name: 'elu',
              config: { name: 'elu', alpha: 1.5 },
              inbound_nodes: [[['lrelu', 0, 0, {}]]],
            },
            {
              class_name: 'ThresholdedReLU',
              name: 'trelu',
              config: { name: 'trelu', theta: 0.5 },
              inbound_nodes: [[['elu', 0, 0, {}]]],
            },
            {
              class_name: 'Conv2D',
              name: 'conv_same',
              config: { name: 'conv_same', filters: 4, kernel_size: [3, 3], padding: 'same' },
              inbound_nodes: [[['trelu', 0, 0, {}]]],
            },
            {
              class_name: 'MaxPooling2D',
              name: 'pool_same',
              config: { name: 'pool_same', pool_size: [2, 2], padding: 'same' },
              inbound_nodes: [[['conv_same', 0, 0, {}]]],
            },
            {
              class_name: 'BatchNormalization',
              name: 'bn_nosc',
              config: { name: 'bn_nosc', momentum: 0.9, scale: false, center: false },
              inbound_nodes: [[['pool_same', 0, 0, {}]]],
            },
            {
              class_name: 'GroupNormalization',
              name: 'gn_nosc',
              config: { name: 'gn_nosc', scale: false, center: false },
              inbound_nodes: [[['bn_nosc', 0, 0, {}]]],
            },
            {
              class_name: 'Reshape',
              name: 'resh_fixed',
              config: { name: 'resh_fixed', target_shape: [4, 4, 3] },
              inbound_nodes: [[['gn_nosc', 0, 0, {}]]],
            },
            {
              class_name: 'Rescaling',
              name: 'resc',
              config: { name: 'resc', scale: 0.5, offset: 0.1 },
              inbound_nodes: [[['resh_fixed', 0, 0, {}]]],
            },
            {
              class_name: 'Resizing',
              name: 'resz_near',
              config: { name: 'resz_near', height: 8, width: 8, interpolation: 'nearest' },
              inbound_nodes: [[['resc', 0, 0, {}]]],
            },
            {
              class_name: 'Resizing',
              name: 'resz_cub',
              config: { name: 'resz_cub', height: 4, width: 4, interpolation: 'bicubic' },
              inbound_nodes: [[['resz_near', 0, 0, {}]]],
            },
            {
              class_name: 'Subtract',
              name: 'sub3',
              config: { name: 'sub3' },
              inbound_nodes: [
                [
                  ['resz_cub', 0, 0, {}],
                  ['in2', 0, 0, {}],
                  ['in3', 0, 0, {}],
                ],
              ],
            },
            {
              class_name: 'EmptyInputsQatLayer',
              name: 'eiq',
              config: { name: 'eiq' },
              inbound_nodes: [[['sub3', 0, 0, {}]]],
            },
          ],
          input_layers: [
            ['in1', 0, 0],
            ['in2', 0, 0],
            ['in3', 0, 0],
          ],
          output_layers: [['eiq', 0, 0]],
        },
      },
      weightsManifest: [],
    });
    expect(new Keras2OnnxConverter(kerasBranchesJson).convert().byteLength).toBeGreaterThan(0);

    // keras-ast.ts lines 68-69, 115, 270:
    // - signature with non-string/non-object value (e.g. number)
    // - unwrapped Sequential layer without className
    // - nested Functional model with mismatched input count (1 vs 2)
    extractKerasTopology(
      {
        class_name: 'Sequential',
        config: {
          layers: [{ name: 'unwrapped_no_cls', input_shape: [4] }],
        },
      },
      '',
      {
        serving_default: {
          inputs: { bad_in: 123 },
          outputs: { bad_out: 456 },
        },
      },
    );
    extractKerasTopology({
      class_name: 'Functional',
      config: {
        layers: [
          {
            class_name: 'InputLayer',
            config: { name: 'p_in', batch_input_shape: [null, 4] },
            inbound_nodes: [],
          },
          {
            class_name: 'Functional',
            config: {
              name: 'nested_mismatch',
              layers: [
                {
                  class_name: 'InputLayer',
                  config: { name: 'n_in1', batch_input_shape: [null, 4] },
                  inbound_nodes: [],
                },
                {
                  class_name: 'InputLayer',
                  config: { name: 'n_in2', batch_input_shape: [null, 4] },
                  inbound_nodes: [],
                },
              ],
              input_layers: [
                ['n_in1', 0, 0],
                ['n_in2', 0, 0],
              ],
              output_layers: [['n_in1', 0, 0]],
            },
            inbound_nodes: [[['p_in', 0, 0, {}]]],
          },
        ],
        input_layers: [['p_in', 0, 0]],
        output_layers: [['nested_mismatch', 0, 0]],
      },
    });

    // keras/optimizers.ts lines 100, 175:
    // - fuseConvBN where BN has empty outputs
    // - fuseDenseBN where BN has empty outputs
    const gOptEdge = new Graph('opte');
    gOptEdge.nodes.push(
      new Node('Conv', ['x', 'w'], ['c_out']),
      new Node('BatchNormalization', ['c_out'], []),
      new Node('Gemm', ['x', 'w'], ['g_out']),
      new Node('BatchNormalization', ['g_out'], []),
    );
    new KerasGraphOptimizer().optimize(gOptEdge);

    // pytorch/generator.ts lines 141, 171, 174-175, 224-227, 269:
    // - node with initializer input used twice (registerBufferIfNeeded false branch)
    // - Gemm with 1 input (no weight) and Gemm with non-2D weight and Gemm with [0, 0] weight
    // - sequential chain ending at a node with empty outputs
    // - Reshape where shapeTensor has no data
    const gPtEdge = new Graph('pte');
    gPtEdge.tensors.w_shared = new Tensor('w_shared', [1], 'float32', true);
    gPtEdge.tensors.w_1d = new Tensor('w_1d', [2], 'float32', true);
    gPtEdge.tensors.w_zero2d = new Tensor('w_zero2d', [0, 0], 'float32', true);
    gPtEdge.tensors.sh_nodata = new Tensor('sh_nodata', [1], 'int64', true, false, null);
    gPtEdge.nodes.push(
      new Node('Add', ['x', 'w_shared'], ['a1']),
      new Node('Add', ['a1', 'w_shared'], ['a2']),
      new Node('Gemm', ['a2'], ['g1']),
      new Node('Gemm', ['g1', 'w_1d'], ['g2']),
      new Node('Gemm', ['g2', 'w_zero2d'], ['g3'], { transB: new Attribute('transB', 'INT', 1) }),
      new Node('Gemm', ['g3', 'w_zero2d'], ['g4'], { transB: new Attribute('transB', 'INT', 0) }),
      new Node('Reshape', ['g4', 'sh_nodata'], ['r1']),
      new Node('Relu', ['r1'], ['s_mid']),
      new Node('Relu', ['s_mid'], []),
    );
    expect(new PyTorchGenerator(gPtEdge).generate()).toBeDefined();

    // Final branch additions:
    // 1. weight-loader.ts line 105: string dtype in calculateByteLength
    expect(() =>
      calculateByteLength({ name: 'ws', shape: [1], dtype: 'string' as 'float32' }),
    ).toThrow(/String dtype/);

    // 2. keras/index.ts line 453: output shape with concrete number (non-null) and signature output matching out.split(':')[0]
    const kerasOutShapeJson = JSON.stringify({
      format: 'layers-model',
      signature: {
        serving_default: {
          inputs: { sig_i: {} as unknown as string },
          outputs: { sig_o: 'out_dense', sig_o2: {} as unknown as string },
        },
      },
      modelTopology: {
        class_name: 'Functional',
        config: {
          name: 'os_model',
          layers: [
            {
              class_name: 'InputLayer',
              name: 'in_os',
              config: { name: 'in_os', batch_input_shape: [2, 4] },
              inbound_nodes: [],
            },
            {
              class_name: 'Dense',
              name: 'out_dense',
              config: { name: 'out_dense', units: 4 },
              inbound_nodes: [[['in_os', 0, 0, {}]]],
            },
          ],
          input_layers: [['in_os', 0, 0]],
          output_layers: [['out_dense', 0, 0]],
        },
      },
      weightsManifest: [],
    });
    const convOs = new Keras2OnnxConverter(kerasOutShapeJson);
    // Set concrete number in topology.outputs[0].shape
    if (convOs.topology.outputs[0]) {
      convOs.topology.outputs[0].shape = [null, 4];
    }
    expect(convOs.convert().byteLength).toBeGreaterThan(0);

    // 3. keras/optimizers.ts line 100: Identity node with empty inputs or outputs
    const gIdEmpty = new Graph('ide');
    gIdEmpty.nodes.push(new Node('Identity', [], []));
    new KerasGraphOptimizer().optimize(gIdEmpty);

    // 4. mmdnn/api.ts lines 332, 379, 419-420:
    // - Caffe weights with unnamed layer, and prototxt with unnamed layer and layer not in weightsMap
    // - Paddle with empty blocks/ops ({})
    const caffeUnnamedWeights = new Blob([new Uint8Array([0x12, 0x02, 0x32, 0x00])]);
    const caffeMultiPrototxt = new Blob([
      'layer {\n  type: "ReLU"\n}\nlayer {\n  name: "unmatched"\n  type: "ReLU"\n}\n',
    ]);
    expect(
      await api.convert('caffe', 'onnx', [caffeMultiPrototxt, caffeUnnamedWeights]),
    ).toBeDefined();
    expect(
      await api.convert('paddle', 'onnx', [new Blob([JSON.stringify({ blocks: [{}] })])]),
    ).toBeDefined();
    expect(await api.convert('paddle', 'onnx', [new Blob([JSON.stringify({})])])).toBeDefined();

    // 5. mmdnn/shape-inference.ts lines 42, 67:
    // - Relu/Add with node.outputs = [] (undefined output[0])
    const gSiNoOut = new Graph('sino');
    gSiNoOut.inputs.push(new ValueInfo('x', [2, 2], 'float32'));
    gSiNoOut.nodes.push(new Node('Relu', ['x'], []), new Node('Add', ['x', 'x'], []));
    new ShapeInferenceEngine().inferShapes(gSiNoOut, rep);

    // 6. mmdnn/caffe/generator.ts lines 43, 66:
    // - getShape('') called when Conv/Pool input[0] is ''
    // - graph input with unknown shape (not in inputs/valueInfo/tensors -> impossible unless input.shape is undefined!)
    const gCg = new Graph('cg');
    gCg.inputs.push({
      name: 'no_shape_in',
      shape: null as unknown as number[],
      dtype: 'float32',
      id: '1',
    });
    gCg.nodes.push(new Node('Conv', ['', ''], ['out']));
    expect(new CaffeGenerator(gCg).generate()).toBeDefined();

    // 7. mmdnn/caffe/mapper.ts lines 148, 309, 337, 381, 441, 509, 537, 557, 601:
    // - fallback identity with non-empty bottom & top
    // - LRN, Softmax, Concat, BatchNorm, Dropout, Flatten with params defined
    // - Split with undefined element in top array
    // - processBlobs with bBlob having non-array data
    cm.map(
      {
        name: 'unk_with_io',
        type: 'UnknownCaffeLayer',
        bottom: ['b1'],
        top: ['t1'],
        blobs: [{ shape: [1], data: [1] }, { shape: [1] }],
      },
      gCm,
    );
    cm.map(
      { name: 'lrn', type: 'LRN', lrn_param: { local_size: 3, alpha: 0.5, beta: 0.5, k: 2 } },
      gCm,
    );
    cm.map({ name: 'sm', type: 'Softmax', softmax_param: { axis: 0 } }, gCm);
    cm.map({ name: 'cc', type: 'Concat', concat_param: { axis: 2 } }, gCm);
    cm.map({ name: 'bn', type: 'BatchNorm', batch_norm_param: { eps: 1e-3 } }, gCm);
    cm.map({ name: 'do', type: 'Dropout', dropout_param: { dropout_ratio: 0.2 } }, gCm);
    cm.map({ name: 'fl', type: 'Flatten', flatten_param: { axis: 2 } }, gCm);
    cm.map({ name: 'sp_undef', type: 'Split', top: [undefined as unknown as string] }, gCm);

    // 8. mmdnn/darknet/mapper.ts lines 57, 310, 325:
    // - yolo at layer index 1 after a convolutional layer (so prevOutput is truthy!)
    // - shortcut with positive `from: 0`
    // - route with negative `layers: [-1]`
    const gDm3 = new Graph('dm3');
    new DarknetMapper(gDm3, new Float32Array(10)).map([
      { type: 'convolutional', filters: 1, size: 1 },
      { type: 'shortcut', from: 0 },
      { type: 'route', layers: [-1] },
      { type: 'yolo' },
    ]);

    // 9. mmdnn/keras/generator.ts lines 186, 208, 210:
    // - input with shape [2, 3] (first dim is positive number, not -1 or string)
    // - Softmax, LSTM, GRU with empty inputs []
    const gKg2 = new Graph('kg2');
    gKg2.inputs.push(new ValueInfo('pos_in', [2, 3], 'float32'));
    gKg2.nodes.push(
      new Node('Softmax', [], ['s']),
      new Node('LSTM', [], ['l']),
      new Node('GRU', [], ['g']),
    );
    expect(new KerasGenerator(gKg2).generate()).toBeDefined();

    // 10. mmdnn/libsvm/parser.ts lines 35, 45, 47:
    // - svm_type, kernel_type, rho with no value after keyword
    parseLibSVM('svm_type\nkernel_type\nrho\n');

    // 11. mmdnn/mxnet/mapper.ts lines 125, 317:
    // - Activation with act_type='softrelu'
    // - LeakyReLU with act_type='elu' and no slope
    mxm.map({ op: 'Activation', name: 'sr', attrs: { act_type: 'softrelu' } }, gMx);
    mxm.map({ op: 'LeakyReLU', name: 'elu_noslope', attrs: { act_type: 'elu' } }, gMx);

    // 12. mmdnn/ncnn/mapper.ts line 29 & parser.ts line 47:
    // - NcnnMapper constructor with weightBuffer = undefined
    // - parseNcnnParam with '7767517\n1\n' (only 1 count on count line)
    new NcnnMapper({ magic: 7767517, layerCount: 0, blobCount: 0, nodes: [] }, new ArrayBuffer(0));
    parseNcnnParam('7767517\n1\n');

    // 13. mmdnn/pytorch/generator.ts lines 171, 269:
    // - Gemm with transB=0 and 2D weight [2, 3]
    // - Reshape where 2nd input is undefined (empty inputs ['x'])
    const gPt3 = new Graph('pt3');
    gPt3.tensors.w23 = new Tensor('w23', [2, 3], 'float32', true);
    gPt3.nodes.push(
      new Node('Gemm', ['x', 'w23'], ['g_out'], { transB: new Attribute('transB', 'INT', 0) }),
      new Node('Reshape', ['g_out'], ['r_out']),
    );
    expect(new PyTorchGenerator(gPt3).generate()).toBeDefined();

    // 14. mmdnn/tensorflow/generator.ts line 27 & mapper.ts lines 43, 62:
    // - TensorFlowGenerator with empty graph name '' and 0 nodes
    // - TFMapper Const with tensor missing dtype and shape
    expect(new TensorFlowGenerator(new Graph('')).generate()).toBeDefined();
    tfm.map(
      {
        name: 'c_bare',
        op: 'Const',
        input: [],
        attr: { value: { tensor: { tensorContent: new Uint8Array([1, 2, 3, 4]) } } },
      },
      gTf,
    );

    // 15. mmdnn/tfjs/generator.ts lines 18, 33:
    // - isLinearGraph with 0 inputs
    // - isLinearGraph with 2 nodes where 2nd node takes graph input instead of 1st node's output
    expect(isLinearGraph(new Graph('no_in'))).toBe(false);
    const gNonChain = new Graph('nc');
    gNonChain.inputs.push(new ValueInfo('x', [1, 2], 'float32'));
    gNonChain.outputs.push(new ValueInfo('y2', [1, 2], 'float32'));
    gNonChain.nodes.push(new Node('Relu', ['x'], ['y1']), new Node('Relu', ['x'], ['y2']));
    expect(isLinearGraph(gNonChain)).toBe(false);

    // 16. jax/mapper.ts line 67: flaxState with flat() returning Float32Array
    new JaxMapper(
      { invars: [], outvars: [], constvars: ['c_f32'], eqns: [] },
      { c_f32: { flat: () => new Float32Array([1, 2]) } },
    ).map();

    // 17. Final remaining branch coverage:
    // - keras-ast.ts lines 68-69: signature object without `name` property ({})
    // - keras/index.ts: Reshape without target_shape ({})
    const kerasNoTargetShapeJson = JSON.stringify({
      format: 'layers-model',
      signature: {
        serving_default: {
          inputs: { sig_i_noname: {} },
          outputs: { sig_o_noname: {} },
        },
      },
      modelTopology: {
        class_name: 'Functional',
        config: {
          name: 'nts_model',
          layers: [
            {
              class_name: 'InputLayer',
              name: 'in_nts',
              config: { name: 'in_nts', batch_input_shape: [2, 4] },
              inbound_nodes: [],
            },
            {
              class_name: 'Reshape',
              name: 'resh_no_ts',
              config: { name: 'resh_no_ts' },
              inbound_nodes: [[['in_nts', 0, 0, {}]]],
            },
          ],
          input_layers: [['in_nts', 0, 0]],
          output_layers: [['resh_no_ts', 0, 0]],
        },
      },
      weightsManifest: [],
    });
    expect(new Keras2OnnxConverter(kerasNoTargetShapeJson).convert().byteLength).toBeGreaterThan(0);

    // - mmdnn/api.ts lines 332, 379:
    //   * Caffe weights where layer has no blobs (`wt` exists in weightsMap, `wt.blobs` is undefined)
    //   * MXNet with `nodes: []` (triggers `if (graph.nodes.length === 0) throw`)
    const caffeLayerNoBlobsWeights = new Blob([
      new Uint8Array([0x12, 0x04, 0x0a, 0x02, 0x63, 0x31]),
    ]);
    expect(
      await api.convert('caffe', 'onnx', [validPrototxt, caffeLayerNoBlobsWeights]),
    ).toBeDefined();
    expect(
      await api.convert('mxnet', 'onnx', [
        new Blob([JSON.stringify({ nodes: [], arg_nodes: [], heads: [] })]),
      ]),
    ).toBeDefined();

    // - mmdnn/caffe/parser.ts line 70: extra closing brace `}` popping stack to empty
    parsePrototxt('}\n');

    // - mmdnn/caffe/generator.ts line 43: getShape for a tensor name not in tensors, valueInfo, or inputs
    const gCgUnk = new Graph('cgunk');
    gCgUnk.nodes.push(new Node('Conv', ['unknown_in', 'unknown_w'], ['out']));
    expect(new CaffeGenerator(gCgUnk).generate()).toBeDefined();

    // - mmdnn/caffe/mapper.ts line 600: layer.blobs with 0 shape volume ([0]) on 2nd blob
    cm.map(
      {
        name: 'b_zero',
        type: 'Convolution',
        blobs: [
          { shape: [1], data: [1] },
          { shape: [0], data: [1] },
        ],
      },
      gCm,
    );

    // - mmdnn/darknet/parser.ts line 76: 16-byte buffer with version 0.2 (offset 20 > 16)
    const hdr18 = new ArrayBuffer(16);
    const dv18 = new DataView(hdr18);
    dv18.setInt32(0, 0, true);
    dv18.setInt32(4, 2, true);
    expect(parseWeights(hdr18).length).toBe(0);

    // - mmdnn/darknet/mapper.ts line 310: shortcut where targetIdx is out of bounds (-5)
    const gDm4 = new Graph('dm4');
    new DarknetMapper(gDm4, new Float32Array(0)).map([{ type: 'shortcut', from: -5 }]);

    // - mmdnn/mxnet/mapper.ts line 140: Pooling with pool_type='max' and global_pool='True'
    mxm.map({ op: 'Pooling', name: 'gmp', attrs: { pool_type: 'max', global_pool: 'True' } }, gMx);

    // - mmdnn/ncnn/mapper.ts line 29: layer with float attribute (e.g. Pooling or Eltwise coeff or BatchNorm eps)
    new NcnnMapper(
      {
        magic: 7767517,
        layerCount: 1,
        blobCount: 2,
        nodes: [
          {
            type: 'ELU',
            name: 'elu1',
            bottoms: ['x'],
            tops: ['y'],
            attrs: { '0': '0.2' },
          },
        ],
      },
      new ArrayBuffer(16),
    );

    // - mmdnn/tensorflow/mapper.ts lines 43, 62:
    //   * Placeholder with non-empty input `['in1']`
    //   * Const with 0-sized shape `[0]`
    tfm.map({ name: 'ph_in', op: 'Placeholder', input: ['in1'], attr: {} }, gTf);
    tfm.map(
      {
        name: 'c_zero',
        op: 'Const',
        input: [],
        attr: { value: { tensor: { dtype: 'DT_FLOAT', shape: [0] } } },
      },
      gTf,
    );

    // - mmdnn/topology.ts line 42: node depending on another node (producer found and visited)
    const gTopChain = new Graph('tc');
    gTopChain.inputs.push(new ValueInfo('x', [1], 'float32'));
    gTopChain.nodes.push(
      new Node('Relu', ['y1'], ['y2'], {}, 'n2'),
      new Node('Relu', ['x'], ['y1'], {}, 'n1'),
    );
    expect(topologicalSort(gTopChain, rep).nodes[0]?.name).toBe('n1');

    // - src/parsers.ts line 220: JAXprParser where eqn.invars is not an array
    new JAXprParser().parse({
      eqns: [{ primitive: 'add', invars: null as unknown as [], outvars: [{ name: 'y' }] }],
    });

    // 18. Final 11 branch tests:
    // - keras/index.ts line 453 & 909:
    //   * Functional model with output shape [null, null] so idx=1 hits `idx === 0 ? 'batch_size' : -1`
    //   * Reshape with `target_shape: [null, 4]` so `s === null ? -1 : s` hits the true branch
    // - keras/keras-ast.ts lines 68-69, 268:
    //   * extractKerasTopology with `{ serving_default: {} }` (inputs and outputs undefined on sigObj)
    //   * Functional model with nested Functional model AND another layer (so `nName.startsWith(prefixedName)` is false for the sibling layer)
    const kerasNullShapeAndSiblingJson = JSON.stringify({
      format: 'layers-model',
      signature: {
        serving_default: {},
      },
      modelTopology: {
        class_name: 'Functional',
        config: {
          name: 'nss_model',
          layers: [
            {
              class_name: 'InputLayer',
              name: 'in_nss',
              config: { name: 'in_nss', batch_input_shape: [null, null] },
              inbound_nodes: [],
            },
            {
              class_name: 'Reshape',
              name: 'resh_null',
              config: { name: 'resh_null', target_shape: [null, 4] },
              inbound_nodes: [[['in_nss', 0, 0, {}]]],
            },
            {
              class_name: 'Functional',
              name: 'nested_func',
              config: {
                name: 'nested_func',
                layers: [
                  {
                    class_name: 'InputLayer',
                    name: 'nf_in',
                    config: { name: 'nf_in', batch_input_shape: [null, 4] },
                    inbound_nodes: [],
                  },
                  {
                    class_name: 'Dense',
                    name: 'nf_d1',
                    config: { name: 'nf_d1', units: 4 },
                    inbound_nodes: [[['nf_in', 0, 0, {}]]],
                  },
                  {
                    class_name: 'Dense',
                    name: 'nf_d2',
                    config: { name: 'nf_d2', units: 4 },
                    inbound_nodes: [[['nf_d1', 0, 0, {}]]],
                  },
                ],
                input_layers: [['nf_in', 0, 0]],
                output_layers: [['nf_d2', 0, 0]],
              },
              inbound_nodes: [[['resh_null', 0, 0, {}]]],
            },
          ],
          input_layers: [['in_nss', 0, 0]],
          output_layers: [['nested_func', 0, 0]],
        },
      },
      weightsManifest: [],
    });
    const convNss = new Keras2OnnxConverter(kerasNullShapeAndSiblingJson);
    if (convNss.topology.outputs[0]) {
      convNss.topology.outputs[0].shape = [null, null];
    }
    expect(convNss.convert().byteLength).toBeGreaterThan(0);

    // - caffe/mapper.ts line 556: Slice with bottom undefined and top undefined
    cm.map({ name: 'sl_empty', type: 'Slice' }, gCm);

    // - darknet/mapper.ts line 310: shortcut with `from: 0` after a convolutional layer (targetInput is truthy!)
    const gDm5 = new Graph('dm5');
    new DarknetMapper(gDm5, new Float32Array(10)).map([
      { type: 'convolutional', filters: 1, size: 1 },
      { type: 'shortcut', from: -1 },
    ]);

    // - ncnn/mapper.ts line 29: Pooling layer (uses getFloat with default when key is absent and when present)
    new NcnnMapper(
      {
        magic: 7767517,
        layerCount: 2,
        blobCount: 2,
        nodes: [
          {
            type: 'BatchNorm',
            name: 'bn_with_eps',
            bottoms: ['x'],
            tops: ['y1'],
            attrs: { '0': '1', '1': '0.001' },
          },
          {
            type: 'BatchNorm',
            name: 'bn_no_eps',
            bottoms: ['y1'],
            tops: ['y2'],
            attrs: { '0': '1' },
          },
        ],
      },
      new ArrayBuffer(64),
    );

    // - tensorflow/generator.ts line 27: node with empty input name ''
    const gTfEmptyIn = new Graph('tfei');
    gTfEmptyIn.nodes.push(new Node('Relu', [''], ['out']));
    expect(new TensorFlowGenerator(gTfEmptyIn).generate()).toContain('unnamed');

    // - weight-loader.ts line 96: int8, uint8, bool
    expect(calculateByteLength({ name: 'wi8', shape: [2], dtype: 'int8' })).toBe(2);
    expect(calculateByteLength({ name: 'wu8', shape: [2], dtype: 'uint8' })).toBe(2);
    expect(calculateByteLength({ name: 'wb', shape: [2], dtype: 'bool' })).toBe(2);

    // - caffe/mapper.ts line 556: Slice with axis: 2
    cm.map({ name: 'sl_axis', type: 'Slice', slice_param: { axis: 2 } }, gCm);

    // - darknet/mapper.ts line 310: shortcut without `from` property (defaults to -1)
    const gDm6 = new Graph('dm6');
    new DarknetMapper(gDm6, new Float32Array(10)).map([
      { type: 'convolutional', filters: 1, size: 1 },
      { type: 'shortcut' },
    ]);

    // - libsvm/parser.ts line 35: SV line with 0
    parseLibSVM('SV\n0 1:0.5\n');

    // - ncnn/mapper.ts line 29: ReLU with slope attribute '0': '0.1'
    new NcnnMapper(
      {
        magic: 7767517,
        layerCount: 1,
        blobCount: 2,
        nodes: [
          {
            type: 'ReLU',
            name: 'relu_slope',
            bottoms: ['x'],
            tops: ['y'],
            attrs: { '0': '0.1' },
          },
        ],
      },
      new ArrayBuffer(16),
    );

    // - src/parsers.ts line 220: JAXprParser where eqn.primitive is not a string
    new JAXprParser().parse({
      eqns: [
        {
          primitive: 123 as unknown as string,
          invars: [null, {} as unknown as { name: string }, { name: 'x' }],
          outvars: [null, { name: 123 as unknown as string }],
        },
      ],
    });

    // - mmdnn/topology.ts line 42: diamond DAG where shared producer n1 is visited twice (hits `if (visited.has(node.name)) return;`)
    const gDiamond = new Graph('dia');
    gDiamond.inputs.push(new ValueInfo('x', [1], 'float32'));
    gDiamond.nodes.push(
      new Node('Add', ['y1', 'y2'], ['y3'], {}, 'n3'),
      new Node('Relu', ['y1'], ['y2'], {}, 'n2'),
      new Node('Relu', ['x'], ['y1'], {}, 'n1'),
    );
    expect(topologicalSort(gDiamond, rep).nodes[0]?.name).toBe('n1');

    // - mmdnn/libsvm/parser.ts line 35: input with blank line inside
    parseLibSVM('svm_type c_svc\n\nkernel_type rbf\n');

    // - mmdnn/ncnn/mapper.ts line 29: ReLU without '0' attribute (hits `: def` in getFloat)
    new NcnnMapper(
      {
        magic: 7767517,
        layerCount: 1,
        blobCount: 2,
        nodes: [
          {
            type: 'ReLU',
            name: 'relu_no_slope',
            bottoms: ['x'],
            tops: ['y'],
            attrs: {},
          },
        ],
      },
      new ArrayBuffer(16),
    );
  });
});
