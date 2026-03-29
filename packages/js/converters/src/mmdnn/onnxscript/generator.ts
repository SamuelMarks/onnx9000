/* eslint-disable */
// @ts-nocheck
import { Graph, Node } from '@onnx9000/core';

export class OnnxScriptGenerator {
  graph: Graph;

  constructor(graph: Graph) {
    this.graph = graph;
  }

  private sanitize(name: string): string {
    if (!name) return 'unnamed';
    let sanitized = name.replace(/[^a-zA-Z0-9_]/g, '_');
    if (/^[0-9]/.test(sanitized)) {
      sanitized = 'v_' + sanitized;
    }
    return sanitized;
  }

  generate(): string {
    const lines = [
      'import onnxscript',
      'from onnxscript import opset15 as op',
      'from onnxscript import FLOAT',
      '',
      '@onnxscript.script()',
    ];

    const knownVars = new Set<string>();
    const requiredInputs = new Set<string>();

    for (const node of this.graph.nodes) {
      for (const i of node.inputs) {
        if (!i) continue;
        const s = this.sanitize(i);
        if (!knownVars.has(s)) {
          requiredInputs.add(s);
        }
      }
      for (const o of node.outputs) {
        if (!o) continue;
        knownVars.add(this.sanitize(o));
      }
    }

    const inputArgsList = Array.from(requiredInputs).map((name) => `${name}: FLOAT[...]`);
    if (inputArgsList.length === 0) {
      inputArgsList.push('input: FLOAT[...]');
    }
    const sigArgs = inputArgsList.join(', ');

    lines.push(`def model(${sigArgs}):`);

    for (const node of this.graph.nodes) {
      const outNames = node.outputs.map((o) => this.sanitize(o)).join(', ');
      const inNames = node.inputs.map((i) => this.sanitize(i)).join(', ');
      const opName = node.opType;

      let attrs = '';
      if (node.attributes) {
        const attrList = [];
        for (const [k, v] of Object.entries(node.attributes)) {
          let valStr = '';
          if (typeof v.value === 'string') {
            valStr = `"${v.value}"`;
          } else if (Array.isArray(v.value)) {
            valStr = `[${v.value.join(', ')}]`;
          } else {
            valStr = String(v.value);
          }
          attrList.push(`${k}=${valStr}`);
        }
        if (attrList.length > 0) {
          attrs = `, ${attrList.join(', ')}`;
        }
      }

      if (inNames.length > 0) {
        lines.push(`    ${outNames} = op.${opName}(${inNames}${attrs})`);
      } else {
        lines.push(`    ${outNames} = op.${opName}(${attrs.substring(2)})`);
      }
    }

    const outputNames = this.graph.outputs.map((o) => this.sanitize(o.name)).join(', ');
    if (outputNames) {
      lines.push(`    return ${outputNames}`);
    } else {
      lines.push(`    pass`);
    }

    // Generate boilerplate to make it runnable
    lines.push('');
    lines.push('if __name__ == "__main__":');
    lines.push('    onnx_model = model.to_model_proto()');
    lines.push('    print("SUCCESS: ONNXScript model generated correctly")');
    lines.push('    # To save: onnx.save(onnx_model, "model.onnx")');

    return lines.join('\n') + '\n';
  }
}
