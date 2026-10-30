/**
 * @fileoverview OnnxAdapter.ts
 * Provides OnnxAdapter functionality for the Sphinx Demo UI.
 */
export interface VizNode {
  /** Property `id`. */
  id: string;
  /** Property `name`. */
  name: string;
  /** Property `opType`. */
  opType: string;
  /** Property `inputs`. */
  inputs: string[];
  /** Property `outputs`. */
  outputs: string[];
  /** Property `attributes`. */
  attributes?: Record<string, string | number | boolean>;
}

/** Interface `VizGraph` definition. */
export interface VizGraph {
  /** Property `nodes`. */
  nodes: VizNode[];
  /** Property `inputs`. */
  inputs: { name: string; type: string }[];
  /** Property `outputs`. */
  outputs: { name: string; type: string }[];
}

/** Interface `CytoscapeElement` definition. */
export interface CytoscapeElement {
  /** Property `group`. */
  group: 'nodes' | 'edges';
  /** Property `data`. */
  data: object;
  /** Property `classes`. */
  classes?: string;
}

/** Represents the `OnnxAdapter` class. */
export class OnnxAdapter {
  /**
   * Executes `toCytoscape` on `OnnxAdapter`.
   * @param graph Parameter `graph`.
   * @returns Result of `toCytoscape`.
   */
  public static toCytoscape(graph: VizGraph): CytoscapeElement[] {
    const elements: CytoscapeElement[] = [];
    const tensorToSourceMap = new Map<string, string>();

    // 1. Add Graph Inputs
    for (const input of graph.inputs) {
      elements.push({
        group: 'nodes',
        data: {
          id: input.name,
          label: input.name,
          type: 'input',
          dtype: input.type,
        },
        classes: 'onnx-input',
      });
      tensorToSourceMap.set(input.name, input.name);
    }

    // 2. Map Node Outputs to Tensor Names
    for (const node of graph.nodes) {
      elements.push({
        group: 'nodes',
        data: {
          id: node.id,
          label: node.opType,
          type: 'operator',
          name: node.name,
          attributes: node.attributes,
        },
        classes: 'onnx-node',
      });
      for (const out of node.outputs) {
        tensorToSourceMap.set(out, node.id);
      }
    }

    // 3. Create Edges
    for (const node of graph.nodes) {
      for (const inputName of node.inputs) {
        let sourceId = tensorToSourceMap.get(inputName);

        // Initializer / Constant fallback
        if (!sourceId) {
          sourceId = `init-${inputName}`;
          tensorToSourceMap.set(inputName, sourceId);
          elements.push({
            group: 'nodes',
            data: { id: sourceId, label: inputName, type: 'initializer' },
            classes: 'onnx-initializer',
          });
        }

        elements.push({
          group: 'edges',
          data: {
            id: `edge-${sourceId}-${node.id}-${inputName}`,
            source: sourceId,
            target: node.id,
            label: inputName,
          },
        });
      }
    }

    // 4. Add Graph Outputs
    for (const output of graph.outputs) {
      const sourceId = tensorToSourceMap.get(output.name);
      if (sourceId) {
        elements.push({
          group: 'nodes',
          data: {
            id: `output-${output.name}`,
            label: output.name,
            type: 'output',
            dtype: output.type,
          },
          classes: 'onnx-output',
        });

        elements.push({
          group: 'edges',
          data: {
            id: `edge-${sourceId}-output-${output.name}`,
            source: sourceId,
            target: `output-${output.name}`,
            label: output.name,
          },
        });
      }
    }

    return elements;
  }
}
