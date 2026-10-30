/**
 * @fileoverview keras-ast.ts
 * Provides keras-ast functionality for the converters package.
 */
import { ConversionError } from '@onnx9000/core';
import type { JsonArray, JsonObject } from './tfjs-parser.js';

/** Interface `KerasTensorSpec` definition. */
export interface KerasTensorSpec {
  /** Property `name`. */
  name: string; // e.g. "layer_name:node_index:tensor_index"
  /** Property `shape`. */
  shape: (number | null)[];
  /** Property `dtype`. */
  dtype: string;
}

/** Interface `KerasNodeSpec` definition. */
export interface KerasNodeSpec {
  /** Property `className`. */
  className: string;
  /** Property `name`. */
  name: string; // unique node name like 'layer_name:node_index'
  /** Property `layerName`. */
  layerName: string; // original Keras layer name (used for looking up weights)
  /** Property `inboundNodes`. */
  inboundNodes: string[]; // tensor names this node consumes, e.g., 'source_layer:0:0'
  /** Property `config`. */
  config: JsonObject;
  /** Property `nodeIndex`. */
  nodeIndex: number;
}

/** Interface `KerasModelTopology` definition. */
export interface KerasModelTopology {
  /** Property `inputs`. */
  inputs: KerasTensorSpec[];
  /** Property `outputs`. */
  outputs: KerasTensorSpec[];
  /** Property `nodes`. */
  nodes: Map<string, KerasNodeSpec>; // Key is node name e.g. 'layer_name:0'
  /** Property `signatures`. */
  signatures?: Record<string, { inputs: Record<string, string>; outputs: Record<string, string> }>; // e.g. "serving_default"
}

/**
 * Executes `extractKerasTopology`.
 * @param modelConfig Parameter `modelConfig`.
 * @param parentPrefix Parameter `parentPrefix`.
 * @param rawSignatures Parameter `rawSignatures`.
 * @returns Result of `extractKerasTopology`.
 */
export function extractKerasTopology(
  modelConfig: JsonObject,
  parentPrefix: string = '',
  rawSignatures?: JsonObject,
): KerasModelTopology {
  const topology: KerasModelTopology = {
    inputs: [],
    outputs: [],
    nodes: new Map<string, KerasNodeSpec>(),
  };

  if (rawSignatures && !parentPrefix) {
    topology.signatures = {};
    for (const [sigName, sigDef] of Object.entries(rawSignatures)) {
      const sigObj = sigDef as JsonObject;
      const inputsObj = (sigObj.inputs as JsonObject) || {};
      const outputsObj = (sigObj.outputs as JsonObject) || {};

      const parsedInputs: Record<string, string> = {};
      for (const [k, v] of Object.entries(inputsObj)) {
        if (typeof v === 'object' && v !== null && (v as JsonObject).name) {
          parsedInputs[k] = (v as JsonObject).name as string;
        } else if (typeof v === 'string') {
          parsedInputs[k] = v;
        }
      }

      const parsedOutputs: Record<string, string> = {};
      for (const [k, v] of Object.entries(outputsObj)) {
        if (typeof v === 'object' && v !== null && (v as JsonObject).name) {
          parsedOutputs[k] = (v as JsonObject).name as string;
        } else if (typeof v === 'string') {
          parsedOutputs[k] = v;
        }
      }

      topology.signatures[sigName] = {
        inputs: parsedInputs,
        outputs: parsedOutputs,
      };
    }
  }

  const className = modelConfig.class_name as string;
  const config = modelConfig.config as JsonObject;

  // To keep track of hoisted nested models outputs for wiring
  const nestedModelOutputMap = new Map<string, string>();

  if (className === 'Sequential') {
    const layersList = config.layers as JsonArray;
    let prevLayerName: string | undefined;

    for (const layerVal of layersList) {
      const layerObj = layerVal as JsonObject;
      let lClassName = layerObj.class_name as string;
      let lConfig = layerObj.config as JsonObject;
      let lName = lConfig?.name as string;

      // Handle Sequential layers where class_name / config are not wrapped
      if (!lClassName && typeof layerObj.name === 'string') {
        lName = layerObj.name;
        lClassName = (layerObj.className as string) || 'Unknown';
        lConfig = layerObj;
      }

      const prefixedName = parentPrefix ? `${parentPrefix}/${lName}` : lName;
      const inboundNodes: string[] = prevLayerName
        ? [`${parentPrefix ? `${parentPrefix}/${prevLayerName}` : prevLayerName}:0:0`]
        : [];

      // Identify first layer's input
      if (prevLayerName === undefined) {
        let shapeArray: JsonArray | undefined;
        if (lConfig.batch_input_shape) {
          shapeArray = lConfig.batch_input_shape as JsonArray;
        } else if (lConfig.input_shape) {
          shapeArray = [null, ...(lConfig.input_shape as JsonArray)];
        }

        const shape = shapeArray ? shapeArray.map((s) => (typeof s === 'number' ? s : null)) : [];
        const dtype = typeof lConfig.dtype === 'string' ? lConfig.dtype : 'float32';

        topology.inputs.push({
          name: `${prefixedName}_input:0:0`,
          shape,
          dtype,
        });

        // If the first layer isn't explicitly an InputLayer, we still feed it the synthetic input
        if (lClassName !== 'InputLayer') {
          inboundNodes.push(`${prefixedName}_input:0:0`);
        }
      }

      if (lClassName === 'Functional' || lClassName === 'Sequential' || lClassName === 'Model') {
        const nestedTopology = extractKerasTopology(layerObj, prefixedName);

        // Hoist nested nodes
        for (const [nName, nSpec] of nestedTopology.nodes.entries()) {
          topology.nodes.set(nName, nSpec);
        }

        // Bridge connections. For sequential, we just connect the output of the previous layer to the input of the nested model
        // We know sequential nested models only have 1 input.
        if (nestedTopology.inputs.length > 0 && inboundNodes.length > 0) {
          const nestedInput = nestedTopology.inputs[0] as KerasTensorSpec;
          for (const [nName, nSpec] of topology.nodes.entries()) {
            if (nName.startsWith(prefixedName)) {
              nSpec.inboundNodes = nSpec.inboundNodes.map((inNode: string) =>
                inNode === nestedInput.name ? (inboundNodes[0] as string) : inNode,
              );
            }
          }
          const baseInputName = (nestedInput.name.split(':')[0] as string).replace(/_input$/, '');
          const internalInputNodeName = `${baseInputName}:0`;
          const internalInputNode = topology.nodes.get(internalInputNodeName);
          if (internalInputNode?.className === 'InputLayer') {
            topology.nodes.delete(internalInputNodeName);
          }
        }

        // Update prevLayerName to point to the last layer of the nested model
        if (nestedTopology.outputs.length > 0) {
          prevLayerName = (nestedTopology.outputs[0] as KerasTensorSpec).name.split(
            ':',
          )[0] as string;
          // Strip parentPrefix if it was added, as prevLayerName is used in the loop
          if (parentPrefix && prevLayerName.startsWith(`${parentPrefix}/`)) {
            prevLayerName = prevLayerName.substring(parentPrefix.length + 1);
          }
        }
      } else {
        topology.nodes.set(`${prefixedName}:0`, {
          className: lClassName,
          name: `${prefixedName}:0`,
          layerName: lName,
          nodeIndex: 0,
          inboundNodes,
          config: lConfig,
        });
        prevLayerName = lName;
      }
    }

    // Output of Sequential is the last layer
    if (prevLayerName) {
      topology.outputs.push({
        name: `${parentPrefix ? `${parentPrefix}/${prevLayerName}` : prevLayerName}:0:0`,
        shape: [null, -1],
        dtype: 'float32',
      });
    }
  } else if (className === 'Functional' || className === 'Model') {
    const layersList = config.layers as JsonArray;
    const inputList = config.input_layers as JsonArray;
    const outputList = config.output_layers as JsonArray;

    for (const layerVal of layersList) {
      const layerObj = layerVal as JsonObject;
      const lClassName = layerObj.class_name as string;
      const lConfig = layerObj.config as JsonObject;
      const lName = layerObj.name as string;
      const prefixedName = parentPrefix ? `${parentPrefix}/${lName}` : lName;

      const inboundNodesRaw = layerObj.inbound_nodes as JsonArray | undefined;
      const inboundNodesBase: string[][] = [];

      if (inboundNodesRaw && inboundNodesRaw.length > 0) {
        for (let nodeIndex = 0; nodeIndex < inboundNodesRaw.length; nodeIndex++) {
          const nodeGrpArr = inboundNodesRaw[nodeIndex] as JsonArray;
          const inputs: string[] = [];

          for (const nodeInfo of nodeGrpArr) {
            const nodeInfoArr = nodeInfo as JsonArray;
            const srcLayer = nodeInfoArr[0] as string;
            const srcNodeIndex = nodeInfoArr[1] as number;
            const srcTensorIndex = nodeInfoArr[2] as number;
            const prefixedSrcLayer = parentPrefix ? `${parentPrefix}/${srcLayer}` : srcLayer;
            inputs.push(`${prefixedSrcLayer}:${srcNodeIndex}:${srcTensorIndex}`);
          }
          inboundNodesBase.push(inputs);
        }
      } else {
        inboundNodesBase.push([]);
      }

      if (lClassName === 'Functional' || lClassName === 'Sequential' || lClassName === 'Model') {
        const nestedTopology = extractKerasTopology(layerObj, prefixedName);

        for (const [nName, nSpec] of nestedTopology.nodes.entries()) {
          topology.nodes.set(nName, nSpec);
        }

        // Store mapping: parent output name -> nested model's actual output name
        if (nestedTopology.outputs.length > 0) {
          // Assume 1:1 output mapping for simplicity in Sequential/Functional nests
          // where outputs[0] is the main output
          nestedModelOutputMap.set(
            `${prefixedName}:0:0`,
            (nestedTopology.outputs[0] as KerasTensorSpec).name,
          );
        }

        // Map each inbound node group (execution instance of the nested model)
        for (const instanceInboundNodes of inboundNodesBase) {
          // For a functional model, it might have multiple inputs.
          // instanceInboundNodes maps 1:1 with nestedTopology.inputs
          if (nestedTopology.inputs.length === instanceInboundNodes.length) {
            for (const [inIdx, nestedInput] of nestedTopology.inputs.entries()) {
              const nestedInputName = nestedInput.name; // e.g. nested_model/nested_in:0:0
              const nestedInputBaseName = `${nestedInputName.split(':')[0]}:0`; // nested_model/nested_in:0

              // Re-wire
              for (const [nName, nSpec] of topology.nodes.entries()) {
                if (nName.startsWith(prefixedName)) {
                  nSpec.inboundNodes = nSpec.inboundNodes.map((inN: string) =>
                    inN === nestedInputName ? (instanceInboundNodes[inIdx] as string) : inN,
                  );
                }
              }
              topology.nodes.delete(nestedInputBaseName);
            }
          }
        }
      } else {
        for (const [nodeIndex, currentInbound] of inboundNodesBase.entries()) {
          topology.nodes.set(`${prefixedName}:${nodeIndex}`, {
            className: lClassName,
            name: `${prefixedName}:${nodeIndex}`,
            layerName: lName, // Keep original name for weight lookup
            nodeIndex,
            inboundNodes: currentInbound,
            config: lConfig,
          });
        }
      }
    }

    if (inputList && !parentPrefix) {
      for (const inLayer of inputList) {
        const inArr = inLayer as JsonArray;
        const inName = inArr[0] as string;
        const inNodeIndex = inArr[1] as number;
        const inTensorIndex = inArr[2] as number;

        let shape: (number | null)[] = [];
        let dtype = 'float32';

        const layerNode = topology.nodes.get(`${inName}:0`);
        if (layerNode && layerNode.className === 'InputLayer') {
          if (layerNode.config.batch_input_shape) {
            const shapeArray = layerNode.config.batch_input_shape as JsonArray;
            shape = shapeArray.map((s) => (typeof s === 'number' ? s : null));
          }
          if (typeof layerNode.config.dtype === 'string') {
            dtype = layerNode.config.dtype;
          }
        }

        topology.inputs.push({
          name: `${inName}:${inNodeIndex}:${inTensorIndex}`,
          shape,
          dtype,
        });
      }
    } else if (inputList && parentPrefix) {
      for (const inLayer of inputList) {
        const inArr = inLayer as JsonArray;
        const inName = inArr[0] as string;
        const inNodeIndex = inArr[1] as number;
        const inTensorIndex = inArr[2] as number;
        topology.inputs.push({
          name: `${parentPrefix}/${inName}:${inNodeIndex}:${inTensorIndex}`,
          shape: [],
          dtype: 'float32',
        });
      }
    }

    if (outputList) {
      for (const outLayer of outputList) {
        const outArr = outLayer as JsonArray;
        const outName = outArr[0] as string;
        const outNodeIndex = outArr[1] as number;
        const outTensorIndex = outArr[2] as number;

        let finalOutName = `${outName}:${outNodeIndex}:${outTensorIndex}`;
        if (parentPrefix) {
          finalOutName = `${parentPrefix}/${finalOutName}`;
        }

        // If an output points to a nested model (which was hoisted and no longer exists as a node),
        // we must rewire this output to the actual output of the nested model.
        // We look for a node with the same name as the output prefix to verify it's not missing.
        // If it's missing, it was likely a nested model that was hoisted.
        // To accurately track this without global state, we can scan the hoisted topology nodes
        // that begin with `finalOutName.split(':')[0] + '/'` and see if they were outputs...
        // But an easier way is to just push the name as is, and let the outer call or final pass resolve aliases.

        topology.outputs.push({
          name: finalOutName,
          shape: [], // Inferred later
          dtype: 'float32',
        });
      }
    }

    // Alias Resolution Pass:
    // If a model output points to a Functional layer, it must be re-wired to the Functional layer's output.
    for (const outItem of topology.outputs) {
      const mapped = nestedModelOutputMap.get(outItem.name);
      if (mapped !== undefined) {
        outItem.name = mapped;
      }
    }
  } else {
    throw new ConversionError(`Unsupported root model class: ${className}`);
  }

  return topology;
}
