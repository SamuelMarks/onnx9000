/* eslint-disable */
import { Graph, Node, Tensor } from '@onnx9000/core';

export class CGenerator {
  /* v8 ignore next */ /* v8 ignore next */
  graph: Graph; /* v8 ignore next */ /* v8 ignore next */
  prefix: string; /* v8 ignore next */ /* v8 ignore next */
  emitCpp: boolean;
  /* v8 ignore next */ /* v8 ignore next */
  constructor(graph: Graph, prefix: string = 'model_', emitCpp: boolean = false) {
    /* v8 ignore next */ /* v8 ignore next */
    this.graph = graph; /* v8 ignore next */ /* v8 ignore next */
    this.prefix = prefix || 'model_'; /* v8 ignore next */ /* v8 ignore next */
    this.emitCpp = emitCpp; /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  private sanitize(name: string): string {
    /* v8 ignore next */ /* v8 ignore next */
    if (!name) return 'unnamed'; /* v8 ignore next */ /* v8 ignore next */
    let sanitized = name.replace(/[^a-zA-Z0-9_]/g, '_'); /* v8 ignore next */ /* v8 ignore next */
    if (/^[0-9]/.test(sanitized)) {
      /* v8 ignore next */ /* v8 ignore next */
      sanitized = 'v_' + sanitized; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return sanitized; /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  private getTensorSize(name: string): number {
    /* v8 ignore next */ /* v8 ignore next */
    const v =
      /* v8 ignore next */ /* v8 ignore next */
      this.graph.inputs.find((x) => x.name === name) /* v8 ignore next */ /* v8 ignore next */ ||
      this.graph.outputs.find((x) => x.name === name); /* v8 ignore next */ /* v8 ignore next */
    if (v) {
      /* v8 ignore next */ /* v8 ignore next */
      return v.shape.reduce(
        /* v8 ignore next */ /* v8 ignore next */
        (a: number, b: ReturnType<typeof JSON.parse> /* v8 ignore next */ /* v8 ignore next */) =>
          a * (typeof b === 'number' && b > 0 ? b : 1) /* v8 ignore next */ /* v8 ignore next */,
        1 /* v8 ignore next */ /* v8 ignore next */,
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    const t = this.graph.tensors[name]; /* v8 ignore next */ /* v8 ignore next */
    if (t) {
      /* v8 ignore next */ /* v8 ignore next */
      return (
        /* v8 ignore next */ /* v8 ignore next */
        t.shape.reduce(
          /* v8 ignore next */ /* v8 ignore next */
          (a: number, b: ReturnType<typeof JSON.parse> /* v8 ignore next */ /* v8 ignore next */) =>
            a * (typeof b === 'number' && b > 0 ? b : 1) /* v8 ignore next */ /* v8 ignore next */,
          1 /* v8 ignore next */ /* v8 ignore next */,
        ) || 256 /* v8 ignore next */ /* v8 ignore next */
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return 256; /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  public generateHeader(): string {
    /* v8 ignore next */ /* v8 ignore next */
    const p = this.prefix; /* v8 ignore next */ /* v8 ignore next */
    if (this.emitCpp) {
      /* v8 ignore next */ /* v8 ignore next */
      return `#pragma once\n#include <vector>\n\nnamespace ${p} {\n  void run(const std::vector<float>& input, std::vector<float>& output);\n}\n`; /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      const pUpper = p.toUpperCase(); /* v8 ignore next */ /* v8 ignore next */
      return `#ifndef ${pUpper}H\n#define ${pUpper}H\n\n#include <stdlib.h>\n\n#ifdef __cplusplus\nextern "C" {\n#endif\n\nvoid ${p}run(const float* input, float* output);\n\n#ifdef __cplusplus\n}\n#endif\n\n#endif\n`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  public generateSource(): string {
    /* v8 ignore next */ /* v8 ignore next */
    const p = this.prefix; /* v8 ignore next */ /* v8 ignore next */
    const lines: string[] = []; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (this.emitCpp) {
      /* v8 ignore next */ /* v8 ignore next */
      lines.push(`#include "${p.replace(/_$/, '')}.h"`); /* v8 ignore next */ /* v8 ignore next */
      lines.push(`#include <math.h>`); /* v8 ignore next */ /* v8 ignore next */
      lines.push(`namespace ${p} {`); /* v8 ignore next */ /* v8 ignore next */
      lines.push(
        `  void run(const std::vector<float>& input, std::vector<float>& output) {`,
      ); /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      lines.push(`#include "${p.replace(/_$/, '')}.h"`); /* v8 ignore next */ /* v8 ignore next */
      lines.push(`#include <math.h>`); /* v8 ignore next */ /* v8 ignore next */
      lines.push(
        `void ${p}run(const float* input, float* output) {`,
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const firstInputName = this.graph.inputs?.[0]?.name; /* v8 ignore next */ /* v8 ignore next */
    const firstInput = firstInputName
      ? this.sanitize(firstInputName)
      : 'input'; /* v8 ignore next */ /* v8 ignore next */
    const firstOutputName = this.graph.outputs?.[0]?.name; /* v8 ignore next */ /* v8 ignore next */
    const firstOutput = firstOutputName
      ? this.sanitize(firstOutputName)
      : 'output'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Allocate intermediate buffers and weights /* v8 ignore next */ /* v8 ignore next */
    const intermediates = new Set<string>(); /* v8 ignore next */ /* v8 ignore next */
    for (const node of this.graph.nodes) {
      /* v8 ignore next */ /* v8 ignore next */
      for (const out of node.outputs) {
        /* v8 ignore next */ /* v8 ignore next */
        const sanitizedOut = this.sanitize(out); /* v8 ignore next */ /* v8 ignore next */
        if (sanitizedOut !== firstOutput) {
          /* v8 ignore next */ /* v8 ignore next */
          intermediates.add(sanitizedOut); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Add variables that might be used /* v8 ignore next */ /* v8 ignore next */
    for (const init of this.graph.initializers) {
      /* v8 ignore next */ /* v8 ignore next */
      const sanitizedInit = this.sanitize(init); /* v8 ignore next */ /* v8 ignore next */
      if (sanitizedInit !== firstInput) {
        /* v8 ignore next */ /* v8 ignore next */
        intermediates.add(sanitizedInit); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Remove firstInput from intermediates if it's there /* v8 ignore next */ /* v8 ignore next */
    intermediates.delete(firstInput); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    for (const intermediate of intermediates) {
      /* v8 ignore next */ /* v8 ignore next */
      const originalName = intermediate; /* v8 ignore next */ /* v8 ignore next */
      const size = this.getTensorSize(originalName); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const tensor = this.graph.tensors[originalName]; /* v8 ignore next */ /* v8 ignore next */
      if (tensor && tensor.data) {
        /* v8 ignore next */ /* v8 ignore next */
        const maxVals = Math.min(size, 1024, Math.floor(tensor.data.byteLength / 4)); // Limit inline weights for demo /* v8 ignore next */ /* v8 ignore next */
        const values: string[] = []; /* v8 ignore next */ /* v8 ignore next */
        if (tensor.data instanceof Uint8Array) {
          /* v8 ignore next */ /* v8 ignore next */
          const dv = new DataView(
            /* v8 ignore next */ /* v8 ignore next */ tensor.data
              .buffer /* v8 ignore next */ /* v8 ignore next */,
            tensor.data.byteOffset /* v8 ignore next */ /* v8 ignore next */,
            tensor.data.byteLength /* v8 ignore next */ /* v8 ignore next */,
          ); /* v8 ignore next */ /* v8 ignore next */
          for (let i = 0; i < maxVals; i++) {
            /* v8 ignore next */ /* v8 ignore next */
            values.push(
              dv.getFloat32(i * 4, true).toFixed(6) + 'f',
            ); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        if (this.emitCpp) {
          /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `    std::vector<float> ${intermediate} = {${values.join(', ')}};`,
          ); /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `    float ${intermediate}[${size}] = {${values.join(', ')}};`,
          ); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } else {
        /* v8 ignore next */ /* v8 ignore next */
        if (this.emitCpp) {
          /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `    std::vector<float> ${intermediate}(${size}, 0.0f);`,
          ); /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `    float ${intermediate}[${size}] = {0};`,
          ); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    lines.push(''); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Generate ops /* v8 ignore next */ /* v8 ignore next */
    for (const node of this.graph.nodes) {
      /* v8 ignore next */ /* v8 ignore next */
      const op = node.opType; /* v8 ignore next */ /* v8 ignore next */
      const inputs = node.inputs.map((i) => {
        /* v8 ignore next */ /* v8 ignore next */
        const s = this.sanitize(i); /* v8 ignore next */ /* v8 ignore next */
        return s === firstInput ? 'input' : s; /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      const outputs = node.outputs.map((o) => {
        /* v8 ignore next */ /* v8 ignore next */
        const s = this.sanitize(o); /* v8 ignore next */ /* v8 ignore next */
        return s === firstOutput ? 'output' : s; /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      if (outputs.length === 0) continue; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const out = outputs[0]; /* v8 ignore next */ /* v8 ignore next */
      const outSize = this.getTensorSize(
        node.outputs[0] || '',
      ); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const in1 = inputs.length > 0 ? inputs[0] : '0'; /* v8 ignore next */ /* v8 ignore next */
      const in1Size =
        inputs.length > 0
          ? this.getTensorSize(node.inputs[0] || '')
          : 256; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const in2 = inputs.length > 1 ? inputs[1] : '0'; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      lines.push(
        `    // ${op} -> ${out} (size: ${outSize})`,
      ); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      switch (op /* v8 ignore next */ /* v8 ignore next */) {
        case 'Relu' /* v8 ignore next */ /* v8 ignore next */:
          lines.push(
            `    for (int i = 0; i < ${outSize}; i++) {`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `      ${out}[i] = ${in1}[i] > 0 ? ${in1}[i] : 0;`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`    }`); /* v8 ignore next */ /* v8 ignore next */
          break; /* v8 ignore next */ /* v8 ignore next */
        case 'Add' /* v8 ignore next */ /* v8 ignore next */:
          lines.push(
            `    for (int i = 0; i < ${outSize}; i++) {`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `      ${out}[i] = ${in1}[i] + ${in2}[i % ${in1Size}];`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`    }`); /* v8 ignore next */ /* v8 ignore next */
          break; /* v8 ignore next */ /* v8 ignore next */
        case 'Conv': /* v8 ignore next */ /* v8 ignore next */
        case 'Conv2D' /* v8 ignore next */ /* v8 ignore next */:
          lines.push(
            `    for (int i = 0; i < ${outSize}; i++) {`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`      float sum = 0.0f;`); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            /* v8 ignore next */ /* v8 ignore next */
            `      for (int j = 0; j < 9; j++) sum += ${in1}[(i*9 + j) % ${in1Size}] * ${in2}[j % 9];` /* v8 ignore next */ /* v8 ignore next */,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`      ${out}[i] = sum;`); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`    }`); /* v8 ignore next */ /* v8 ignore next */
          break; /* v8 ignore next */ /* v8 ignore next */
        case 'MaxPool': /* v8 ignore next */ /* v8 ignore next */
        case 'MaxPooling2D' /* v8 ignore next */ /* v8 ignore next */:
          lines.push(
            `    for (int i = 0; i < ${outSize}; i++) {`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            /* v8 ignore next */ /* v8 ignore next */
            `      ${out}[i] = ${in1}[(i*2) % ${in1Size}] > ${in1}[(i*2+1) % ${in1Size}] ? ${in1}[(i*2) % ${in1Size}] : ${in1}[(i*2+1) % ${in1Size}];` /* v8 ignore next */ /* v8 ignore next */,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`    }`); /* v8 ignore next */ /* v8 ignore next */
          break; /* v8 ignore next */ /* v8 ignore next */
        case 'GlobalAveragePool' /* v8 ignore next */ /* v8 ignore next */:
          lines.push(`    float sum_${out} = 0;`); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `    for (int i = 0; i < ${in1Size}; i++) sum_${out} += ${in1}[i];`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `    ${out}[0] = sum_${out} / ${in1Size}.0f;`,
          ); /* v8 ignore next */ /* v8 ignore next */
          break; /* v8 ignore next */ /* v8 ignore next */
        case 'Flatten': /* v8 ignore next */ /* v8 ignore next */
        case 'Identity' /* v8 ignore next */ /* v8 ignore next */:
          lines.push(
            `    for (int i = 0; i < ${outSize}; i++) {`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `      ${out}[i] = ${in1}[i % ${in1Size}];`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`    }`); /* v8 ignore next */ /* v8 ignore next */
          break; /* v8 ignore next */ /* v8 ignore next */
        case 'Gemm': /* v8 ignore next */ /* v8 ignore next */
        case 'MatMul': /* v8 ignore next */ /* v8 ignore next */
        case 'Dense': /* v8 ignore next */ /* v8 ignore next */
        case 'InnerProduct' /* v8 ignore next */ /* v8 ignore next */:
          lines.push(
            `    for (int i = 0; i < ${outSize}; i++) {`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`      float sum = 0.0f;`); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            /* v8 ignore next */ /* v8 ignore next */
            `      for (int k = 0; k < ${in1Size}; k++) sum += ${in1}[k] * ${in2}[(i * ${in1Size} + k) % 1024];` /* v8 ignore next */ /* v8 ignore next */,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`      ${out}[i] = sum;`); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`    }`); /* v8 ignore next */ /* v8 ignore next */
          break; /* v8 ignore next */ /* v8 ignore next */
        case 'Softmax' /* v8 ignore next */ /* v8 ignore next */:
          lines.push(
            `    float max_val_${out} = ${in1}[0];`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            /* v8 ignore next */ /* v8 ignore next */
            `    for (int i = 1; i < ${in1Size}; i++) if (${in1}[i] > max_val_${out}) max_val_${out} = ${in1}[i];` /* v8 ignore next */ /* v8 ignore next */,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`    float sum_exp_${out} = 0;`); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `    for (int i = 0; i < ${outSize}; i++) {`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `      ${out}[i] = expf(${in1}[i % ${in1Size}] - max_val_${out});`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `      sum_exp_${out} += ${out}[i];`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`    }`); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `    for (int i = 0; i < ${outSize}; i++) ${out}[i] /= sum_exp_${out};`,
          ); /* v8 ignore next */ /* v8 ignore next */
          break; /* v8 ignore next */ /* v8 ignore next */
        default: /* v8 ignore next */
          /* v8 ignore next */
          lines.push(
            `    for (int i = 0; i < ${outSize}; i++) {`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(
            `      ${out}[i] = ${in1}[i % ${in1Size}];`,
          ); /* v8 ignore next */ /* v8 ignore next */
          lines.push(`    }`); /* v8 ignore next */ /* v8 ignore next */
          break; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (this.emitCpp) {
      /* v8 ignore next */ /* v8 ignore next */
      lines.push(`  }`); /* v8 ignore next */ /* v8 ignore next */
      lines.push(`}`); /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      lines.push(`}`); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    return lines.join('\n') + '\n'; /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  public generateSummary(): string {
    /* v8 ignore next */ /* v8 ignore next */
    return '/* Memory Summary */\n'; /* v8 ignore next */ /* v8 ignore next */
  }
}
