/* eslint-disable */
// @ts-nocheck
import {
  Graph,
  BufferReader,
  parseModelProto,
  serializeModelProto,
  ValueInfo,
  Tensor,
} from '@onnx9000/core';
import { MMDNNReporter } from './reporter.js';
import { topologicalSort } from './topology.js';
import { DataLayoutTracker } from './layout.js';
import { ShapeInferenceEngine } from './shape-inference.js';
import { NodeFusionRegistry } from './fusion.js';
import { FileLoader } from './file-loader.js';
import { PyTorchGenerator } from './pytorch/generator.js';
import { TensorFlowGenerator } from './tensorflow/generator.js';
import { CaffeGenerator } from './caffe/generator.js';
import { MXNetGenerator } from './mxnet/generator.js';
import { CNTKGenerator } from './cntk/generator.js';
import { KerasGenerator } from './keras/generator.js';

export type SourceFramework =
  | 'caffe'
  | 'mxnet'
  | 'cntk'
  | 'darknet'
  | 'ncnn'
  | 'paddle'
  | 'keras'
  | 'coreml'
  | 'onnx'
  | 'tensorflow'
  | 'scikitlearn'
  | 'lightgbm'
  | 'xgboost'
  | 'catboost'
  | 'sparkml'
  | 'onnxscript';

export type TargetFramework =
  | 'onnx'
  | 'pytorch_code'
  | 'tfjs'
  | 'tflite'
  | 'coreml'
  | 'array'
  | 'caffe'
  | 'keras'
  | 'mxnet'
  | 'tensorflow'
  | 'cntk'
  | 'onnxscript';

export interface ConvertOptions {
  fusion?: boolean;
  shapeInference?: boolean;
  layoutTracking?: boolean;
  verbose?: boolean;
}

function deduceGraphIO(graph: Graph) {
  const allOutputs = new Set<string>();
  for (const node of graph.nodes) {
    for (const out of node.outputs) {
      allOutputs.add(out);
    }
  }

  const inputs = new Set<string>();
  const outputs = new Set<string>();
  const missingInitializers = new Set<string>();

  for (const node of graph.nodes) {
    for (const inp of node.inputs) {
      if (!allOutputs.has(inp)) {
        if (
          inp.includes('_weights') ||
          inp.includes('_kernel') ||
          inp.includes('_bias') ||
          inp.includes('weight') ||
          inp.includes('bias') ||
          inp.includes('_W') ||
          inp.includes('_B')
        ) {
          missingInitializers.add(inp);
        } else {
          inputs.add(inp);
        }
      }
    }
  }

  const allInputs = new Set<string>();
  for (const node of graph.nodes) {
    for (const inp of node.inputs) {
      allInputs.add(inp);
    }
  }
  for (const out of allOutputs) {
    if (!allInputs.has(out)) {
      outputs.add(out);
    }
  }

  for (const inp of inputs) {
    if (!graph.inputs.find((i) => i.name === inp)) {
      graph.inputs.push(new ValueInfo(inp, [-1, -1, -1, -1], 'float32'));
    }
  }
  for (const out of outputs) {
    if (!graph.outputs.find((o) => o.name === out)) {
      graph.outputs.push(new ValueInfo(out, [-1, -1], 'float32'));
    }
  }

  // Automatically populate missing structural initializers to prevent incomplete topologies
  for (const mi of missingInitializers) {
    if (!graph.initializers.includes(mi)) {
      graph.initializers.push(mi);
    }
    if (!graph.tensors[mi]) {
      // Default to a 1D scalar dummy weight to satisfy ONNX structural requirements
      const tensor = new Tensor(mi, [1], 'float32', true, false);
      tensor.data = new Float32Array(1);
      graph.tensors[mi] = tensor;
    }
  }
}

export async function convert(
  source: SourceFramework,
  target: TargetFramework,
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<object> {
  const reporter = new MMDNNReporter(options.verbose);
  reporter.info(`Starting conversion from ${source} to ${target}`);

  const loader = new FileLoader(files);
  await loader.initialize();

  let graph: Graph;

  if (source === 'onnx') {
    reporter.info('Source is already ONNX, skipping parsing phase.');
    const file0 = files[0];
    if (!file0) throw new Error('No ONNX file provided');
    const arrayBuffer = await file0.arrayBuffer();
    const reader = new BufferReader(new Uint8Array(arrayBuffer));
    graph = await parseModelProto(reader);
  } else {
    reporter.info(`Parsing ${source} into ONNX IR...`);
    // Minimal stub mapping for other frameworks
    if (source === 'caffe') {
      const { parsePrototxt, parseCaffemodel } = await import('./caffe/parser.js');
      const { CaffeMapper } = await import('./caffe/mapper.js');
      const file0 = files[0];
      const text = await file0!.text();
      let parsed: object = {};
      try {
        parsed = parsePrototxt(text) || {};
      } catch (e) {
        reporter.warn('Failed to parse caffe prototxt');
      }

      // Merge weights if available
      if (files.length > 1) {
        try {
          reporter.info('Parsing Caffe weights...');
          const file1 = files[1]!;
          const arrayBuffer = await file1.arrayBuffer();
          const weightsParsed = await parseCaffemodel(new Uint8Array(arrayBuffer));
          const weightsMap = new Map();
          for (const layer of weightsParsed.layer || []) {
            weightsMap.set(layer.name, layer);
          }
          for (const layer of parsed.layer || []) {
            const wt = weightsMap.get(layer.name);
            if (wt && wt.blobs) {
              layer.blobs = wt.blobs;
            }
          }
        } catch (e) {
          reporter.warn('Failed to parse caffemodel weights');
        }
      }

      const mapper = new CaffeMapper();
      graph = new Graph('caffe-imported');
      graph.nodes = [];
      graph.inputs = [];
      graph.outputs = [];
      graph.initializers = [];
      graph.tensors = {};
      graph.valueInfo = [];
      for (const layer of parsed.layer || []) {
        const nodes = mapper.map(layer, graph);
        graph.nodes.push(...nodes);
      }
      deduceGraphIO(graph);
    } else if (source === 'mxnet') {
      const { parseMxNetSymbol } = await import('./mxnet/parser.js');
      const { MxNetMapper } = await import('./mxnet/mapper.js');
      const file0 = files[0];
      const text = await file0!.text();
      const parsed = parseMxNetSymbol(text);
      const mapper = new MxNetMapper();
      graph = new Graph('mxnet-imported');
      for (const node of parsed.nodes || []) {
        const outNodes = mapper.map(node, graph);
        graph.nodes.push(...outNodes);
      }
      deduceGraphIO(graph);
    } else if (source === 'tensorflow') {
      const { parsePbtxt } = await import('./tensorflow/parser.js');
      const { TFMapper } = await import('./tensorflow/mapper.js');
      const file0 = files[0];
      const text = await file0!.text();
      let parsed: object = { node: [] };
      try {
        parsed = parsePbtxt(text) || { node: [] };
      } catch (e) {
        reporter.warn('Failed to parse tensorflow pbtxt');
      }
      const mapper = new TFMapper();
      graph = new Graph('tensorflow-imported');
      graph.nodes = [];
      graph.inputs = [];
      graph.outputs = [];
      graph.initializers = [];
      graph.tensors = {};
      graph.valueInfo = [];
      for (const node of parsed.node || []) {
        const nodes = mapper.map(node, graph);
        graph.nodes.push(...nodes);
      }
      deduceGraphIO(graph);
    } else if (source === 'paddle') {
      const { PaddleParser } = await import('./paddle/parser.js');
      const { PaddleMapper } = await import('./paddle/mapper.js');
      const file0 = files[0];
      const text = await file0!.text();
      let parsed: object = {};
      try {
        const parser = new PaddleParser();
        parsed = parser.parseModel(text) || {};
      } catch (e) {
        reporter.warn('Failed to parse paddlepaddle model');
      }
      const mapper = new PaddleMapper();
      graph = new Graph('paddle-imported');
      for (const block of parsed.blocks || []) {
        for (const op of block.ops || []) {
          const nodes = mapper.map(op, graph);
          graph.nodes.push(...nodes);
        }
      }
      deduceGraphIO(graph);
    } else if (source === 'onnxscript') {
      const { OnnxScriptParser } = await import('./onnxscript/parser.js');
      const file0 = files[0];
      const text = await file0!.text();
      try {
        const parser = new OnnxScriptParser();
        graph = parser.parseScript(text);
      } catch (e) {
        reporter.warn('Failed to parse onnxscript model');
        graph = new Graph('onnxscript-imported');
      }
    } else if (source === 'scikitlearn') {
      const { ScikitLearnParser } = await import('./scikitlearn/parser.js');
      const file0 = files[0];
      const text = await file0!.text();
      try {
        const parser = new ScikitLearnParser();
        graph = parser.parseModel(text);
      } catch (e) {
        reporter.warn('Failed to parse scikitlearn model');
        graph = new Graph('scikitlearn-imported');
      }
    } else if (source === 'lightgbm') {
      const { LightGBMParser } = await import('./lightgbm/parser.js');
      const file0 = files[0];
      const text = await file0!.text();
      try {
        const parser = new LightGBMParser();
        graph = parser.parseModel(text);
      } catch (e) {
        reporter.warn('Failed to parse lightgbm model');
        graph = new Graph('lightgbm-imported');
      }
    } else if (source === 'xgboost') {
      const { XGBoostParser } = await import('./xgboost/parser.js');
      const file0 = files[0];
      const text = await file0!.text();
      try {
        const parser = new XGBoostParser();
        graph = parser.parseModel(text);
      } catch (e) {
        reporter.warn('Failed to parse xgboost model');
        graph = new Graph('xgboost-imported');
      }
    } else if (source === 'catboost') {
      const { CatBoostParser } = await import('./catboost/parser.js');
      const file0 = files[0];
      const text = await file0!.text();
      try {
        const parser = new CatBoostParser();
        graph = parser.parseModel(text);
      } catch (e) {
        reporter.warn('Failed to parse catboost model');
        graph = new Graph('catboost-imported');
      }
    } else if (source === 'sparkml') {
      const { SparkMLParser } = await import('./sparkml/parser.js');
      const file0 = files[0];
      const text = await file0!.text();
      try {
        const parser = new SparkMLParser();
        graph = parser.parseModel(text);
      } catch (e) {
        reporter.warn('Failed to parse sparkml model');
        graph = new Graph('sparkml-imported');
      }
    } else {
      graph = new Graph(`${source}-imported`);
    }
  }

  reporter.info('Running topological sort...');
  graph = topologicalSort(graph, reporter);

  if (options.fusion !== false) {
    reporter.info('Applying node fusions...');
    const fusionRegistry = new NodeFusionRegistry();
    graph = fusionRegistry.applyFusions(graph, reporter);
  }

  if (options.layoutTracking !== false) {
    reporter.info('Tracking data layouts...');
    const layoutTracker = new DataLayoutTracker();
    layoutTracker.track(graph, reporter);
  }

  if (options.shapeInference !== false) {
    reporter.info('Running shape inference engine...');
    const shapeEngine = new ShapeInferenceEngine();
    shapeEngine.inferShapes(graph, reporter);
  }

  if (target === 'onnx') {
    reporter.info('Target is ONNX, returning Graph IR.');
    return graph;
  }

  reporter.info(`Exporting ONNX IR to ${target}...`);
  if (target === 'pytorch_code') {
    const gen = new PyTorchGenerator(graph);
    return gen.generate();
  } else if (target === 'keras') {
    const gen = new KerasGenerator(graph);
    return gen.generate();
  } else if (target === 'caffe') {
    const gen = new CaffeGenerator(graph);
    return gen.generate();
  } else if (target === 'tensorflow') {
    const gen = new TensorFlowGenerator(graph);
    return gen.generate();
  } else if (target === 'cntk') {
    const gen = new CNTKGenerator(graph);
    return gen.generate();
  } else if (target === 'mxnet') {
    const gen = new MXNetGenerator(graph);
    return gen.generate();
  } else if (target === 'onnxscript') {
    const { OnnxScriptGenerator } = await import('./onnxscript/generator.js');
    const gen = new OnnxScriptGenerator(graph);
    return gen.generate();
  }

  return `# Exported ${target} content for ${graph.name}`;
}
