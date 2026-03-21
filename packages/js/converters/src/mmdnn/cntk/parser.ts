export interface CNTKNode {
  name: string;
  op: string;
  inputs: string[];
  attributes: Record<string, any>;
  uid: string;
}

export interface CNTKModel {
  nodes: CNTKNode[];
  inputs: string[];
  outputs: string[];
}

export class CNTKParser {
  public parse(dictionary: any): CNTKModel {
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
