/**
 * @fileoverview CNTK dictionary model parser for MMDNN.
 */
import type { AttributeValue } from '@onnx9000/core';

/**
 * Intermediate representation of a CNTK computation node.
 */
export interface CNTKNode {
  /** Property `name`. */
  name: string;
  /** Property `op`. */
  op: string;
  /** Property `inputs`. */
  inputs: string[];
  /** Property `attributes`. */
  attributes: Record<string, AttributeValue>;
  /** Property `uid`. */
  uid: string;
}

/**
 * Raw CNTK node entry in a serialized CNTK dictionary.
 */
export interface CNTKRawNode {
  /** Property `name`. */
  name?: string;
  /** Property `op`. */
  op?: string;
  /** Property `inputs`. */
  inputs?: string[];
  /** Property `attributes`. */
  attributes?: Record<string, AttributeValue>;
  /** Property `uid`. */
  uid?: string;
}

/**
 * Raw CNTK dictionary structure.
 */
export interface CNTKDictionary {
  /** Property `nodes`. */
  nodes?: CNTKRawNode[];
  /** Property `inputs`. */
  inputs?: string[];
  /** Property `outputs`. */
  outputs?: string[];
}

/**
 * Parsed CNTK model representation.
 */
export interface CNTKModel {
  /** Property `nodes`. */
  nodes: CNTKNode[];
  /** Property `inputs`. */
  inputs: string[];
  /** Property `outputs`. */
  outputs: string[];
}

/**
 * Parser for CNTK dictionary models.
 */
export class CNTKParser {
  /**
   * Parses a raw CNTK dictionary into a structured `CNTKModel`.
   * @param dictionary Raw CNTK model dictionary.
   * @returns Parsed `CNTKModel`.
   */
  public parse(dictionary: CNTKDictionary): CNTKModel {
    const nodes: CNTKNode[] = [];
    const modelInputs: string[] = [];
    const modelOutputs: string[] = [];

    if (dictionary.inputs && Array.isArray(dictionary.inputs)) {
      modelInputs.push(...dictionary.inputs);
    }
    if (dictionary.outputs && Array.isArray(dictionary.outputs)) {
      modelOutputs.push(...dictionary.outputs);
    }

    if (dictionary.nodes && Array.isArray(dictionary.nodes)) {
      for (const node of dictionary.nodes) {
        nodes.push({
          name: node.name || node.uid || 'unknown',
          op: node.op || 'unknown',
          inputs: node.inputs || [],
          attributes: node.attributes || {},
          uid: node.uid || node.name || 'unknown',
        });
      }
    }

    return { nodes, inputs: modelInputs, outputs: modelOutputs };
  }
}
