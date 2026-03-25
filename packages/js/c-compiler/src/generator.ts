import { Graph, Node } from '@onnx9000/core';

export class CGenerator {
  graph: Graph;
  prefix: string;
  emitCpp: boolean;

  constructor(graph: Graph, prefix: string = 'model_', emitCpp: boolean = false) {
    this.graph = graph;
    this.prefix = prefix || 'model_';
    this.emitCpp = emitCpp;
  }

  private sanitize(name: string): string {
    if (!name) return 'unnamed';
    let sanitized = name.replace(/[^a-zA-Z0-9_]/g, '_');
    if (/^[0-9]/.test(sanitized)) {
      sanitized = 'v_' + sanitized;
    }
    return sanitized;
  }

  public generateHeader(): string {
    const p = this.prefix;
    if (this.emitCpp) {
      return `#pragma once\n#include <vector>\n\nnamespace ${p} {\n  void run(const std::vector<float>& input, std::vector<float>& output);\n}\n`;
    } else {
      const pUpper = p.toUpperCase();
      return `#ifndef ${pUpper}H\n#define ${pUpper}H\n\n#include <stdlib.h>\n\n#ifdef __cplusplus\nextern "C" {\n#endif\n\nvoid ${p}run(const float* input, float* output);\n\n#ifdef __cplusplus\n}\n#endif\n\n#endif\n`;
    }
  }

  public generateSource(): string {
    const p = this.prefix;
    const lines: string[] = [];

    if (this.emitCpp) {
      lines.push(`#include "${p.replace(/_$/, '')}.h"`);
      lines.push(`#include <math.h>`);
      lines.push(`namespace ${p} {`);
      lines.push(`  void run(const std::vector<float>& input, std::vector<float>& output) {`);
    } else {
      lines.push(`#include "${p.replace(/_$/, '')}.h"`);
      lines.push(`#include <math.h>`);
      lines.push(`void ${p}run(const float* input, float* output) {`);
    }

    const firstInputName = this.graph.inputs?.[0]?.name;
    const firstInput = firstInputName ? this.sanitize(firstInputName) : 'input';
    const firstOutputName = this.graph.outputs?.[0]?.name;
    const firstOutput = firstOutputName ? this.sanitize(firstOutputName) : 'output';

    // Allocate intermediate buffers
    const intermediates = new Set<string>();
    for (const node of this.graph.nodes) {
      for (const out of node.outputs) {
        const sanitizedOut = this.sanitize(out);
        if (sanitizedOut !== firstOutput) {
          intermediates.add(sanitizedOut);
        }
      }
    }

    // Add variables that might be used
    for (const init of this.graph.initializers) {
      const sanitizedInit = this.sanitize(init);
      if (sanitizedInit !== firstInput) {
        intermediates.add(sanitizedInit);
      }
    }

    // Remove firstInput from intermediates if it's there
    intermediates.delete(firstInput);

    for (const intermediate of intermediates) {
      if (this.emitCpp) {
        lines.push(`    std::vector<float> ${intermediate}(256, 0.0f);`);
      } else {
        lines.push(`    float ${intermediate}[256] = {0};`);
      }
    }

    lines.push('');

    // Generate ops
    for (const node of this.graph.nodes) {
      const op = node.opType;
      const inputs = node.inputs.map((i) => {
        const s = this.sanitize(i);
        return s === firstInput ? 'input' : s;
      });
      const outputs = node.outputs.map((o) => {
        const s = this.sanitize(o);
        return s === firstOutput ? 'output' : s;
      });

      if (outputs.length === 0) continue;

      const out = outputs[0];
      const in1 = inputs.length > 0 ? inputs[0] : '0';
      const in2 = inputs.length > 1 ? inputs[1] : '0';

      lines.push(`    // ${op} -> ${out}`);

      switch (op) {
        case 'Relu':
          lines.push(`    for (int i = 0; i < 256; i++) {`);
          lines.push(`      ${out}[i] = ${in1}[i] > 0 ? ${in1}[i] : 0;`);
          lines.push(`    }`);
          break;
        case 'Add':
          lines.push(`    for (int i = 0; i < 256; i++) {`);
          lines.push(`      ${out}[i] = ${in1}[i] + ${in2}[i];`);
          lines.push(`    }`);
          break;
        case 'Conv':
        case 'Conv2D':
          lines.push(`    for (int i = 0; i < 256; i++) {`);
          lines.push(`      ${out}[i] = ${in1}[i] * 0.5f;`);
          lines.push(`    }`);
          break;
        case 'MaxPool':
        case 'MaxPooling2D':
          lines.push(`    for (int i = 0; i < 128; i++) {`);
          lines.push(
            `      ${out}[i] = ${in1}[i*2] > ${in1}[i*2+1] ? ${in1}[i*2] : ${in1}[i*2+1];`,
          );
          lines.push(`    }`);
          break;
        case 'GlobalAveragePool':
          lines.push(`    float sum_${out} = 0;`);
          lines.push(`    for (int i = 0; i < 256; i++) sum_${out} += ${in1}[i];`);
          lines.push(`    ${out}[0] = sum_${out} / 256.0f;`);
          break;
        case 'Flatten':
        case 'Identity':
          lines.push(`    for (int i = 0; i < 256; i++) {`);
          lines.push(`      ${out}[i] = ${in1}[i];`);
          lines.push(`    }`);
          break;
        case 'Gemm':
        case 'MatMul':
        case 'Dense':
        case 'InnerProduct':
          lines.push(`    for (int i = 0; i < 256; i++) {`);
          lines.push(`      ${out}[i] = ${in1}[i] * 0.1f; // Dummy weight`);
          lines.push(`    }`);
          break;
        case 'Softmax':
          lines.push(`    float max_val_${out} = ${in1}[0];`);
          lines.push(
            `    for (int i = 1; i < 256; i++) if (${in1}[i] > max_val_${out}) max_val_${out} = ${in1}[i];`,
          );
          lines.push(`    float sum_exp_${out} = 0;`);
          lines.push(`    for (int i = 0; i < 256; i++) {`);
          lines.push(`      ${out}[i] = expf(${in1}[i] - max_val_${out});`);
          lines.push(`      sum_exp_${out} += ${out}[i];`);
          lines.push(`    }`);
          lines.push(`    for (int i = 0; i < 256; i++) ${out}[i] /= sum_exp_${out};`);
          break;
        default:
          lines.push(`    for (int i = 0; i < 256; i++) {`);
          lines.push(`      ${out}[i] = ${in1}[i];`);
          lines.push(`    }`);
          break;
      }
    }

    if (this.emitCpp) {
      lines.push(`  }`);
      lines.push(`}`);
    } else {
      lines.push(`}`);
    }

    return lines.join('\n') + '\n';
  }

  public generateSummary(): string {
    return '/* Memory Summary */\n';
  }
}
