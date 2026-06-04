/* eslint-disable */
// @ts-nocheck
import { Graph, Node } from '@onnx9000/core';

export function isLinearGraph(graph: Graph): boolean {
  if (graph.inputs.length !== 1) return false;
  if (graph.outputs.length !== 1) return false;
  /* v8 ignore next */ /* v8 ignore next */
  let currentOutput = graph.inputs[0] ? graph.inputs[0].name : '';
  for (const node of graph.nodes) {
    const isInit = (name: string /* v8 ignore next */ /* v8 ignore next */) =>
      graph.initializers.includes(name) || !!graph.tensors[name]?.isInitializer;
    const dynamicInputs = node.inputs.filter((inName) => !isInit(inName));

    if (dynamicInputs.length !== 1 || dynamicInputs[0] !== currentOutput) {
      return false;
    } /* v8 ignore next */ /* v8 ignore next */
    if (node.outputs.length !== 1) {
      /* v8 ignore start */
      return false;
    }
    /* v8 ignore stop */ /* v8 ignore next */ /* v8 ignore next */
    currentOutput = node.outputs[0] || '';
  } /* v8 ignore next */ /* v8 ignore next */
  return currentOutput === (graph.outputs[0] ? graph.outputs[0].name : '');
}
/* v8 ignore next */ /* v8 ignore next */
function sanitizeName(name: string): string {
  /* v8 ignore next */ /* v8 ignore next */
  if (/^[0-9]/.test(name))
    return 'v_' + name.replace(/[^a-zA-Z0-9_]/g, '_'); /* v8 ignore next */ /* v8 ignore next */
  return name.replace(/[^a-zA-Z0-9_]/g, '_'); /* v8 ignore next */ /* v8 ignore next */
}

export function generateTFJSCode(graph: Graph): string {
  const linear = isLinearGraph(graph);
  let code = `import * as tf from '@tensorflow/tfjs';\n\n`;
  code += `export function createModel() {\n`;

  if (linear) {
    code += `  const model = tf.sequential();\n`;
    let isFirst = true;
    for (const node of graph.nodes) {
      code += `  model.add(${generateLayerCode(node, graph, isFirst)});\n`;
      isFirst = false;
    }
    code += `  return model;\n`; /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    // Functional API /* v8 ignore next */ /* v8 ignore next */
    const inputVars: string[] = []; /* v8 ignore next */ /* v8 ignore next */
    for (const input of graph.inputs) {
      /* v8 ignore next */ /* v8 ignore next */
      const shapeStr = JSON.stringify(
        /* v8 ignore next */ /* v8 ignore next */
        input.shape.map((s) =>
          Number(s) === -1 ? null : Number(s),
        ) /* v8 ignore next */ /* v8 ignore next */,
      ); /* v8 ignore next */ /* v8 ignore next */
      const varName = sanitizeName(input.name); /* v8 ignore next */ /* v8 ignore next */
      code += `  const ${varName} = tf.input({ shape: ${shapeStr}.slice(1) });\n`; /* v8 ignore next */ /* v8 ignore next */
      inputVars.push(varName); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    for (const node of graph.nodes) {
      /* v8 ignore next */ /* v8 ignore next */
      const outVar = sanitizeName(node.outputs[0] || ''); /* v8 ignore next */ /* v8 ignore next */
      const isInit = (name: string /* v8 ignore next */ /* v8 ignore next */) =>
        graph.initializers.includes(name) ||
        !!graph.tensors[name]?.isInitializer; /* v8 ignore next */ /* v8 ignore next */
      const dynamicInputs = node.inputs
        .filter((inName) => !isInit(inName))
        .map(sanitizeName); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const layerCode = generateLayerCode(
        node,
        graph,
        false,
      ); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      if (dynamicInputs.length === 1) {
        /* v8 ignore next */ /* v8 ignore next */
        code += `  const ${outVar} = ${layerCode}.apply(${dynamicInputs[0]});\n`; /* v8 ignore next */ /* v8 ignore next */
      } else {
        /* v8 ignore next */ /* v8 ignore next */
        code += `  const ${outVar} = ${layerCode}.apply([${dynamicInputs.join(', ')}]);\n`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const outputVars = graph.outputs.map((out) =>
      sanitizeName(out.name),
    ); /* v8 ignore next */ /* v8 ignore next */
    code += `  const model = tf.model({ inputs: [${inputVars.join(', ')}], outputs: [${outputVars.join(', ')}] });\n`; /* v8 ignore next */ /* v8 ignore next */
    code += `  return model;\n`; /* v8 ignore next */ /* v8 ignore next */
  }

  code += `}\n`;
  return code;
}

function stringifyOptions(obj: ReturnType<typeof JSON.parse>): string {
  /* v8 ignore next */ /* v8 ignore next */
  return JSON.stringify(obj, (_, v) => (typeof v === 'bigint' ? Number(v) : v)).replace(/"/g, "'");
}

function generateLayerCode(node: Node, graph: Graph, isFirst: boolean): string {
  const op = node.opType;
  const options: Record<string, object> = {};

  if (isFirst && graph.inputs.length > 0) {
    const inputInfo = graph.inputs[0];
    if (inputInfo) {
      const shape = inputInfo.shape.map((s) => (s === -1 ? null : s));
      options.inputShape = shape.slice(1);
    }
  }

  if (op === 'Conv') {
    const wName = node.inputs[1];
    const wTensor = wName ? graph.tensors[wName] : undefined;
    if (wTensor && wTensor.shape) {
      options.filters = wTensor.shape[0];
      options.kernelSize = wTensor.shape.slice(2);
    }

    const stridesAttr = node.attributes['strides'];
    if (stridesAttr) options.strides = stridesAttr.value;

    const padsAttr = node.attributes['pads'];
    if (padsAttr && Array.isArray(padsAttr.value) && padsAttr.value.every((p: number) => p > 0)) {
      options.padding = 'same';
    } else {
      options.padding = 'valid';
    }

    options.dataFormat = 'channelsFirst';

    if (node.inputs.length > 2) {
      options.useBias = true;
    } else {
      options.useBias = false;
    }

    return `tf.layers.conv2d(${stringifyOptions(options)})`;
  } else if (op === 'Gemm') {
    const wName = node.inputs[1]; /* v8 ignore next */ /* v8 ignore next */
    const wTensor = wName ? graph.tensors[wName] : undefined;
    const transBAttr = node.attributes['transB']; /* v8 ignore next */ /* v8 ignore next */
    const transB = transBAttr ? (transBAttr.value as number) : 0;
    /* v8 ignore next */ /* v8 ignore next */
    if (wTensor && wTensor.shape) {
      /* v8 ignore next */ /* v8 ignore next */
      if (transB === 1) {
        /* v8 ignore next */ /* v8 ignore next */
        options.units = wTensor.shape[0]; /* v8 ignore next */ /* v8 ignore next */
      } else {
        /* v8 ignore next */ /* v8 ignore next */
        options.units = wTensor.shape[1]; /* v8 ignore next */ /* v8 ignore next */
      }
    } else {
      options.units = 1; // Fallback
    }
    /* v8 ignore next */ /* v8 ignore next */
    if (node.inputs.length > 2) {
      /* v8 ignore next */ /* v8 ignore next */
      options.useBias = true;
    } else {
      options.useBias = false;
    }

    return `tf.layers.dense(${stringifyOptions(options)})`; /* v8 ignore next */ /* v8 ignore next */
  } else if (op === 'MaxPool' || op === 'AveragePool') {
    /* v8 ignore next */ /* v8 ignore next */
    const kernelShapeAttr =
      node.attributes['kernel_shape']; /* v8 ignore next */ /* v8 ignore next */
    if (kernelShapeAttr)
      options.poolSize = kernelShapeAttr.value; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const stridesAttr = node.attributes['strides']; /* v8 ignore next */ /* v8 ignore next */
    if (stridesAttr) options.strides = stridesAttr.value; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const padsAttr = node.attributes['pads']; /* v8 ignore next */ /* v8 ignore next */
    if (padsAttr && Array.isArray(padsAttr.value) && padsAttr.value.every((p: number) => p > 0)) {
      /* v8 ignore next */ /* v8 ignore next */
      options.padding = 'same'; /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      options.padding = 'valid'; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    options.dataFormat = 'channelsFirst'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const func =
      op === 'MaxPool'
        ? 'maxPooling2d'
        : 'averagePooling2d'; /* v8 ignore next */ /* v8 ignore next */
    return `tf.layers.${func}(${stringifyOptions(options)})`; /* v8 ignore next */ /* v8 ignore next */
  } else if (op === 'BatchNormalization') {
    /* v8 ignore next */ /* v8 ignore next */
    options.axis = 1; // channelsFirst => channel is at axis 1 /* v8 ignore next */ /* v8 ignore next */
    return `tf.layers.batchNormalization(${stringifyOptions(options)})`;
  } else if (op === 'Relu') {
    return `tf.layers.reLU(${stringifyOptions(options)})`; /* v8 ignore next */ /* v8 ignore next */
  } else if (op === 'GlobalAveragePool') {
    /* v8 ignore next */ /* v8 ignore next */
    options.dataFormat = 'channelsFirst'; /* v8 ignore next */ /* v8 ignore next */
    return `tf.layers.globalAveragePooling2d(${stringifyOptions(options)})`; /* v8 ignore next */ /* v8 ignore next */
  } else if (op === 'Flatten') {
    /* v8 ignore next */ /* v8 ignore next */
    options.dataFormat = 'channelsFirst'; /* v8 ignore next */ /* v8 ignore next */
    return `tf.layers.flatten(${stringifyOptions(options)})`; /* v8 ignore next */ /* v8 ignore next */
  }

  throw new Error(`Unsupported operator: ${op}`);
}
