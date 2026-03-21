import { Graph, Node } from '@onnx9000/core';

export function isLinearGraph(graph: Graph): boolean {
  if (graph.inputs.length !== 1) return false;
  if (graph.outputs.length !== 1) return false;

  let currentOutput = graph.inputs[0] ? graph.inputs[0].name : '';
  for (const node of graph.nodes) {
    const isInit = (name: string) =>
      graph.initializers.includes(name) || !!graph.tensors[name]?.isInitializer;
    const dynamicInputs = node.inputs.filter((inName) => !isInit(inName));

    if (dynamicInputs.length !== 1 || dynamicInputs[0] !== currentOutput) {
      return false;
    }
    if (node.outputs.length !== 1) {
      return false;
    }
    currentOutput = node.outputs[0] || '';
  }
  return currentOutput === (graph.outputs[0] ? graph.outputs[0].name : '');
}

function sanitizeName(name: string): string {
  if (/^[0-9]/.test(name)) return 'v_' + name.replace(/[^a-zA-Z0-9_]/g, '_');
  return name.replace(/[^a-zA-Z0-9_]/g, '_');
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
    code += `  return model;\n`;
  } else {
    // Functional API
    const inputVars: string[] = [];
    for (const input of graph.inputs) {
      const shapeStr = JSON.stringify(input.shape.map((s) => (s === -1 ? null : s)));
      const varName = sanitizeName(input.name);
      code += `  const ${varName} = tf.input({ shape: ${shapeStr}.slice(1) });\n`;
      inputVars.push(varName);
    }

    for (const node of graph.nodes) {
      const outVar = sanitizeName(node.outputs[0] || '');
      const isInit = (name: string) =>
        graph.initializers.includes(name) || !!graph.tensors[name]?.isInitializer;
      const dynamicInputs = node.inputs.filter((inName) => !isInit(inName)).map(sanitizeName);

      const layerCode = generateLayerCode(node, graph, false);

      if (dynamicInputs.length === 1) {
        code += `  const ${outVar} = ${layerCode}.apply(${dynamicInputs[0]});\n`;
      } else {
        code += `  const ${outVar} = ${layerCode}.apply([${dynamicInputs.join(', ')}]);\n`;
      }
    }

    const outputVars = graph.outputs.map((out) => sanitizeName(out.name));
    code += `  const model = tf.model({ inputs: [${inputVars.join(', ')}], outputs: [${outputVars.join(', ')}] });\n`;
    code += `  return model;\n`;
  }

  code += `}\n`;
  return code;
}

function generateLayerCode(node: Node, graph: Graph, isFirst: boolean): string {
  const op = node.opType;
  const options: Record<string, any> = {};

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

    return `tf.layers.conv2d(${JSON.stringify(options).replace(/"/g, "'")})`;
  } else if (op === 'Gemm') {
    const wName = node.inputs[1];
    const wTensor = wName ? graph.tensors[wName] : undefined;
    const transBAttr = node.attributes['transB'];
    const transB = transBAttr ? (transBAttr.value as number) : 0;

    if (wTensor && wTensor.shape) {
      if (transB === 1) {
        options.units = wTensor.shape[0];
      } else {
        options.units = wTensor.shape[1];
      }
    } else {
      options.units = 1; // Fallback
    }

    if (node.inputs.length > 2) {
      options.useBias = true;
    } else {
      options.useBias = false;
    }

    return `tf.layers.dense(${JSON.stringify(options).replace(/"/g, "'")})`;
  } else if (op === 'MaxPool' || op === 'AveragePool') {
    const kernelShapeAttr = node.attributes['kernel_shape'];
    if (kernelShapeAttr) options.poolSize = kernelShapeAttr.value;

    const stridesAttr = node.attributes['strides'];
    if (stridesAttr) options.strides = stridesAttr.value;

    const padsAttr = node.attributes['pads'];
    if (padsAttr && Array.isArray(padsAttr.value) && padsAttr.value.every((p: number) => p > 0)) {
      options.padding = 'same';
    } else {
      options.padding = 'valid';
    }

    options.dataFormat = 'channelsFirst';

    const func = op === 'MaxPool' ? 'maxPooling2d' : 'averagePooling2d';
    return `tf.layers.${func}(${JSON.stringify(options).replace(/"/g, "'")})`;
  } else if (op === 'BatchNormalization') {
    options.axis = 1; // channelsFirst => channel is at axis 1
    return `tf.layers.batchNormalization(${JSON.stringify(options).replace(/"/g, "'")})`;
  } else if (op === 'Relu') {
    return `tf.layers.reLU(${JSON.stringify(options).replace(/"/g, "'")})`;
  } else if (op === 'GlobalAveragePool') {
    options.dataFormat = 'channelsFirst';
    return `tf.layers.globalAveragePooling2d(${JSON.stringify(options).replace(/"/g, "'")})`;
  } else if (op === 'Flatten') {
    options.dataFormat = 'channelsFirst';
    return `tf.layers.flatten(${JSON.stringify(options).replace(/"/g, "'")})`;
  }

  return `/* TODO: unimplemented ${op} */ tf.layers.dense(${JSON.stringify(options).replace(/"/g, "'")})`;
}
