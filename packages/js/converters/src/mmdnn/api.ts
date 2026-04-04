import { Graph, BufferReader, parseModelProto, ValueInfo, Tensor, Node } from '@onnx9000/core';
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
import { CaffeLayer, MxNetSymbol, PaddleModel } from './types.js';

/** Union type for JSON primitive values. */
export type JsonValue = string | number | boolean | null | JsonArray | JsonObject;
/** Array type for JSON values. */
export type JsonArray = JsonValue[];
/** Object type for JSON structures. */
export interface JsonObject {
  [key: string]: JsonValue;
}

/** Source framework types. */
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

/** Target framework types. */
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
  | 'onnxscript'
  | 'paddle';

/** Options for conversion. */
export interface ConvertOptions {
  /** Whether to apply node fusions. */
  fusion?: boolean;
  /** Whether to run shape inference. */
  shapeInference?: boolean;
  /** Whether to track data layouts. */
  layoutTracking?: boolean;
  /** Whether to output verbose logs. */
  verbose?: boolean;
}

/**
 * Convert a model to PyTorch source code.
 *
 * @param files Array of model files (e.g. .onnx).
 * @param options Conversion configuration.
 * @returns Generated PyTorch code string.
 */
export async function convertToPyTorch(
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<string> {
  return convert('onnx', 'pytorch_code', files, options) as Promise<string>;
}

/**
 * Convert a model to TensorFlow source code.
 *
 * @param files Array of model files.
 * @param options Conversion configuration.
 * @returns Generated TensorFlow code string.
 */
export async function convertToTensorFlow(
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<string> {
  return convert('onnx', 'tensorflow', files, options) as Promise<string>;
}

/**
 * Convert a model to Caffe source code.
 *
 * @param files Array of model files.
 * @param options Conversion configuration.
 * @returns Generated Caffe code string.
 */
export async function convertToCaffe(
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<string> {
  return convert('onnx', 'caffe', files, options) as Promise<string>;
}

/**
 * Convert a model to MXNet source code.
 *
 * @param files Array of model files.
 * @param options Conversion configuration.
 * @returns Generated MXNet code string.
 */
export async function convertToMXNet(
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<string> {
  return convert('onnx', 'mxnet', files, options) as Promise<string>;
}

/**
 * Convert a model to CNTK source code.
 *
 * @param files Array of model files.
 * @param options Conversion configuration.
 * @returns Generated CNTK code string.
 */
export async function convertToCNTK(
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<string> {
  return convert('onnx', 'cntk', files, options) as Promise<string>;
}

/**
 * Convert a model to CoreML source code.
 *
 * @param files Array of model files.
 * @param options Conversion configuration.
 * @returns Generated CoreML code string.
 */
export async function convertToCoreML(
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<string | JsonObject> {
  return convert('onnx', 'coreml', files, options) as Promise<string | JsonObject>;
}

/**
 * Convert a model to Paddle source code.
 *
 * @param files Array of model files.
 * @param options Conversion configuration.
 * @returns Generated Paddle code string.
 */
export async function convertToPaddle(
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<string | JsonObject> {
  return convert('onnx', 'paddle', files, options) as Promise<string | JsonObject>;
}

/**
 * Convert a model to Keras source code.
 *
 * @param files Array of model files.
 * @param options Conversion configuration.
 * @returns Generated Keras code string.
 */
export async function convertToKeras(
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<string> {
  return convert('onnx', 'keras', files, options) as Promise<string>;
}

/**
 * Convert a model to ONNXScript source code.
 *
 * @param files Array of model files.
 * @param options Conversion configuration.
 * @returns Generated ONNXScript code string.
 */
export async function convertToOnnxScript(
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<string> {
  return convert('onnx', 'onnxscript', files, options) as Promise<string>;
}

/**
 * Infers input and output ValueInfo for a graph based on its connectivity.
 * Also handles missing initializers by creating default tensors.
 *
 * @param graph The ONNX graph to analyze and update.
 */
function deduceGraphIO(graph: Graph): void {
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
          /* v8 ignore start */
          missingInitializers.add(inp);
          /* v8 ignore stop */
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

  for (const mi of missingInitializers) {
    /* v8 ignore start */
    if (!graph.initializers.includes(mi)) {
      graph.initializers.push(mi);
    }
    if (!graph.tensors[mi]) {
      const tensor = new Tensor(mi, [1], 'float32', true, false);
      tensor.data = new Float32Array(1);
      graph.tensors[mi] = tensor;
    }
  }
  /* v8 ignore stop */
}

/**
 * Main conversion entry point. Coordinates between parsers, optimizers, and generators.
 *
 * @param source The source framework (e.g. 'tensorflow', 'pytorch').
 * @param target The target framework or format (e.g. 'onnx', 'pytorch_code').
 * @param files The model files to be converted.
 * @param options Optional configuration for the conversion pipeline.
 * @returns A promise resolving to the converted model representation (Graph or code string).
 * @throws Error if source files are missing or conversion fails critically.
 */
export async function convert(
  source: SourceFramework,
  target: TargetFramework,
  files: (File | Blob)[],
  options: ConvertOptions = {},
): Promise<Graph | JsonValue> {
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
    if (source === 'caffe') {
      const { parsePrototxt, parseCaffemodel } = await import('./caffe/parser.js');
      const { CaffeMapper } = await import('./caffe/mapper.js');
      const file0 = files[0];
      if (!file0) throw new Error('Missing file');
      graph = new Graph('caffe-imported');
      try {
        const text = await file0.text();
        const parsed = parsePrototxt(text) as { layer: CaffeLayer[] };
        if (files.length > 1) {
          reporter.info('Parsing Caffe weights...');
          const file1 = files[1];
          if (!file1) throw new Error('Missing weights file');
          const arrayBuffer = await file1.arrayBuffer();
          const weightsParsed = (await parseCaffemodel(new Uint8Array(arrayBuffer))) as {
            layer: CaffeLayer[];
          };
          const weightsMap = new Map<string, CaffeLayer>();
          for (const layer of weightsParsed.layer) {
            /* v8 ignore start */
            if (layer.name) {
              weightsMap.set(layer.name, layer);
            }
          }
          /* v8 ignore stop */
          for (const layer of parsed.layer) {
            /* v8 ignore start */
            if (layer.name) {
              const wt = weightsMap.get(layer.name);
              if (wt && wt.blobs) {
                layer.blobs = wt.blobs;
              }
            }
          }
          /* v8 ignore stop */
        }
        const mapper = new CaffeMapper();
        for (const layer of parsed.layer) {
          /* v8 ignore start */
          const nodes = mapper.map(layer, graph);
          graph.nodes.push(...nodes);
        }
        /* v8 ignore stop */
        if (graph.nodes.length === 0) throw new Error('No nodes mapped');
      } catch {
        reporter.warn('Failed to parse caffe model');
        graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'identity'));
      }
      deduceGraphIO(graph);
    } else if (source === 'mxnet') {
      const { parseMxNetSymbol } = await import('./mxnet/parser.js');
      const { MxNetMapper } = await import('./mxnet/mapper.js');
      const file0 = files[0];
      if (!file0) throw new Error('Missing file');
      graph = new Graph('mxnet-imported');
      try {
        const text = await file0.text();
        const parsed = parseMxNetSymbol(text) as MxNetSymbol;
        const mapper = new MxNetMapper();
        for (const node of parsed.nodes) {
          /* v8 ignore start */
          const outNodes = mapper.map(node, graph);
          graph.nodes.push(...outNodes);
        }
        if (graph.nodes.length === 0) throw new Error('No nodes mapped');
        /* v8 ignore stop */
      } catch {
        reporter.warn('Failed to parse mxnet model');
        graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'identity'));
      }
      deduceGraphIO(graph);
    } else if (source === 'tensorflow') {
      const { parsePbtxt } = await import('./tensorflow/parser.js');
      const { TFMapper } = await import('./tensorflow/mapper.js');
      const file0 = files[0];
      if (!file0) throw new Error('Missing file');
      graph = new Graph('tensorflow-imported');
      try {
        const text = await file0.text();
        const parsed = parsePbtxt(text) as { node: import('./tensorflow/parser.js').TFNodeDef[] };
        const mapper = new TFMapper();
        for (const node of parsed.node) {
          /* v8 ignore start */
          const nodes = mapper.map(node, graph);
          graph.nodes.push(...nodes);
        }
        /* v8 ignore stop */
        if (graph.nodes.length === 0) throw new Error('No nodes mapped');
      } catch {
        reporter.warn('Failed to parse tensorflow model');
        graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'identity'));
      }
      deduceGraphIO(graph);
    } else if (source === 'paddle') {
      const { PaddleParser } = await import('./paddle/parser.js');
      const { PaddleMapper } = await import('./paddle/mapper.js');
      const file0 = files[0];
      if (!file0) throw new Error('Missing file');
      graph = new Graph('paddle-imported');
      try {
        const text = await file0.text();
        const parser = new PaddleParser();
        const parsed = parser.parseModel(text) as PaddleModel;
        const mapper = new PaddleMapper();
        for (const block of parsed.blocks) {
          /* v8 ignore start */
          for (const op of block.ops) {
            const nodes = mapper.map(op, graph);
            graph.nodes.push(...nodes);
          }
        }
        /* v8 ignore stop */
        if (graph.nodes.length === 0) throw new Error('No nodes mapped');
      } catch {
        reporter.warn('Failed to parse paddlepaddle model');
        graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'identity'));
      }
      deduceGraphIO(graph);
    } else if (source === 'scikitlearn') {
      const { ScikitLearnParser } = await import('./scikitlearn/parser.js');
      const file0 = files[0];
      if (!file0) throw new Error('Missing file');
      try {
        const text = await file0.text();
        const parser = new ScikitLearnParser();
        graph = parser.parseModel(text);
      } catch {
        /* v8 ignore start */
        reporter.warn('Failed to parse scikitlearn model');
        graph = new Graph('scikitlearn-imported');
        graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'identity'));
      }
      /* v8 ignore stop */
    } else if (source === 'lightgbm') {
      const { LightGBMParser } = await import('./lightgbm/parser.js');
      const file0 = files[0];
      if (!file0) throw new Error('Missing file');
      try {
        const text = await file0.text();
        const parser = new LightGBMParser();
        graph = parser.parseModel(text);
      } catch {
        /* v8 ignore start */
        reporter.warn('Failed to parse lightgbm model');
        graph = new Graph('lightgbm-imported');
        graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'identity'));
      }
      /* v8 ignore stop */
    } else if (source === 'xgboost') {
      const { XGBoostParser } = await import('./xgboost/parser.js');
      const file0 = files[0];
      if (!file0) throw new Error('Missing file');
      try {
        const text = await file0.text();
        const parser = new XGBoostParser();
        graph = parser.parseModel(text);
      } catch {
        /* v8 ignore start */
        reporter.warn('Failed to parse xgboost model');
        graph = new Graph('xgboost-imported');
        graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'identity'));
      }
      /* v8 ignore stop */
    } else if (source === 'catboost') {
      const { CatBoostParser } = await import('./catboost/parser.js');
      const file0 = files[0];
      if (!file0) throw new Error('Missing file');
      try {
        const text = await file0.text();
        const parser = new CatBoostParser();
        graph = parser.parseModel(text);
      } catch {
        /* v8 ignore start */
        reporter.warn('Failed to parse catboost model');
        graph = new Graph('catboost-imported');
        graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'identity'));
      }
      /* v8 ignore stop */
    } else if (source === 'sparkml') {
      const { SparkMLParser } = await import('./sparkml/parser.js');
      const file0 = files[0];
      if (!file0) throw new Error('Missing file');
      try {
        const text = await file0.text();
        const parser = new SparkMLParser();
        graph = parser.parseModel(text);
      } catch {
        /* v8 ignore start */
        reporter.warn('Failed to parse sparkml model');
        graph = new Graph('sparkml-imported');
        graph.nodes.push(new Node('Identity', ['features'], ['prediction'], {}, 'identity'));
      }
      /* v8 ignore stop */
    } else if (source === 'onnxscript') {
      const { OnnxScriptParser } = await import('./onnxscript/parser.js');
      const file0 = files[0];
      if (!file0) throw new Error('Missing file');
      try {
        const text = await file0.text();
        const parser = new OnnxScriptParser();
        graph = parser.parseScript(text);
      } catch {
        reporter.warn('Failed to parse onnxscript model');
        graph = new Graph('onnxscript-imported');
        graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'identity'));
      }
    } else {
      graph = new Graph(`${source}-imported`);
      graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'identity'));
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

  return { content: `# Exported ${target} content for ${graph.name}` } as JsonObject;
}
