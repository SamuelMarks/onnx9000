/* eslint-disable */
// @ts-nocheck
import { JsonObject, JsonArray } from './tfjs-parser.js';

export interface KerasTensorSpec {
  name: string;
  shape: (number | null)[];
  dtype: string;
}

export interface KerasLayerSpec {
  className: string;
  name: string;
  inboundNodes: string[][]; // list of layer names this layer connects to
  config: JsonObject;
}

export interface KerasModelTopology {
  inputs: KerasTensorSpec[];
  outputs: KerasTensorSpec[];
  layers: Map<string, KerasLayerSpec>;
}

export function extractKerasTopology(modelConfig: JsonObject): KerasModelTopology {
  const topology: KerasModelTopology = {
    inputs: [],
    outputs: [],
    layers: new Map<string, KerasLayerSpec>(),
  };

  const className = modelConfig['class_name'] as string;
  const config = modelConfig['config'] as JsonObject;

  if (className === 'Sequential') {
    const layersList = config['layers'] as JsonArray;
    let prevLayerName: string | undefined = undefined;

    for (const layerVal of layersList) {
      const layerObj = layerVal as JsonObject;
      const lClassName = layerObj['class_name'] as string;
      const lConfig = layerObj['config'] as JsonObject;
      const lName = lConfig['name'] as string;

      const inboundNodes = prevLayerName ? [[prevLayerName]] : [];

      // Identify first layer's input
      if (prevLayerName === undefined) {
        // If it's an InputLayer, its config has the batch_input_shape
        let shapeArray: JsonArray | undefined = undefined;
        if (lConfig['batch_input_shape']) {
          shapeArray = lConfig['batch_input_shape'] as JsonArray;
        }
        const shape = shapeArray ? shapeArray.map((s) => (typeof s === 'number' ? s : null)) : [];
        topology.inputs.push({
          name: lName + '_input',
          shape,
          dtype: (lConfig['dtype'] as string) || 'float32',
        });
      }

      topology.layers.set(lName, {
        className: lClassName,
        name: lName,
        inboundNodes,
        config: lConfig,
      });

      prevLayerName = lName;
    }

    // Output of Sequential is the last layer
    if (prevLayerName) {
      topology.outputs.push({
        name: prevLayerName + '_output',
        shape: [], // Typically dynamic or inferred later
        dtype: 'float32',
      });
    }
  } else if (className === 'Functional' || className === 'Model') {
    const layersList = config['layers'] as JsonArray;
    const inputList = config['input_layers'] as JsonArray;
    const outputList = config['output_layers'] as JsonArray;

    for (const layerVal of layersList) {
      const layerObj = layerVal as JsonObject;
      const lClassName = layerObj['class_name'] as string;
      const lConfig = layerObj['config'] as JsonObject;
      const lName = layerObj['name'] as string;

      const inboundNodesRaw = layerObj['inbound_nodes'] as JsonArray;
      const inboundNodes: string[][] = [];

      if (inboundNodesRaw && inboundNodesRaw.length > 0) {
        for (const nodeGrp of inboundNodesRaw) {
          const nodeGrpArr = nodeGrp as JsonArray;
          const inputs: string[] = [];
          for (const nodeInfo of nodeGrpArr) {
            const nodeInfoArr = nodeInfo as JsonArray;
            inputs.push(nodeInfoArr[0] as string); // First element is usually the source layer name
          }
          inboundNodes.push(inputs);
        }
      }

      topology.layers.set(lName, {
        className: lClassName,
        name: lName,
        inboundNodes,
        config: lConfig,
      });
    }

    if (inputList) {
      for (const inLayer of inputList) {
        const inArr = inLayer as JsonArray;
        const inName = inArr[0] as string;
        topology.inputs.push({
          name: inName,
          shape: [], // Will fetch from layer config
          dtype: 'float32',
        });
      }
    }

    if (outputList) {
      for (const outLayer of outputList) {
        const outArr = outLayer as JsonArray;
        const outName = outArr[0] as string;
        topology.outputs.push({
          name: outName,
          shape: [], // Inferred later
          dtype: 'float32',
        });
      }
    }
  } else {
    throw new Error(`Unsupported root model class: ${className}`);
  }

  // Refine shapes from InputLayers
  for (const input of topology.inputs) {
    const layer = topology.layers.get(input.name);
    if (layer && layer.className === 'InputLayer') {
      if (layer.config['batch_input_shape']) {
        const shapeArray = layer.config['batch_input_shape'] as JsonArray;
        input.shape = shapeArray.map((s) => (typeof s === 'number' ? s : null));
      }
      if (layer.config['dtype']) {
        input.dtype = layer.config['dtype'] as string;
      }
    }
  }

  return topology;
}
