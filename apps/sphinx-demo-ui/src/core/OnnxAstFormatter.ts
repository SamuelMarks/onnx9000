import { VizGraph } from './OnnxAdapter';

/**
 * Utility to format an ONNX AST.
 */
export class OnnxAstFormatter {
  /**
   * Converts a VizGraph to a formatted AST string.
   * @param graph The VizGraph to format.
   * @returns A string representing the graph.
   */
  public static format(graph: VizGraph): string {
    let output = '// ONNX AST Structure:\n';
    output += '// ir_version: 8\n';
    output += '// producer_name: "onnx9000-converter"\n';
    output += 'graph {\n';

    for (const node of graph.nodes) {
      output += '  node {\n';
      for (const input of node.inputs) {
        output += `    input: "${input}"\n`;
      }
      for (const out of node.outputs) {
        output += `    output: "${out}"\n`;
      }
      output += `    op_type: "${node.opType}"\n`;
      if (node.name) {
        output += `    name: "${node.name}"\n`;
      }
      if (node.attributes && Object.keys(node.attributes).length > 0) {
        for (const [key, value] of Object.entries(node.attributes)) {
          output += `    attribute {\n`;
          output += `      name: "${key}"\n`;
          output += `      value: ${JSON.stringify(value)}\n`;
          output += `    }\n`;
        }
      }
      output += '  }\n';
    }

    if (graph.inputs && graph.inputs.length > 0) {
      for (const input of graph.inputs) {
        output += '  input {\n';
        output += `    name: "${input.name}"\n`;
        output += `    type: "${input.type}"\n`;
        output += '  }\n';
      }
    }

    if (graph.outputs && graph.outputs.length > 0) {
      for (const out of graph.outputs) {
        output += '  output {\n';
        output += `    name: "${out.name}"\n`;
        output += `    type: "${out.type}"\n`;
        output += '  }\n';
      }
    }

    output += '}\n';
    return output;
  }
}
