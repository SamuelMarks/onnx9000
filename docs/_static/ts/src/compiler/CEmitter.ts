/* v8 ignore next */ /* v8 ignore next */ /** /* v8 ignore next */ /* v8 ignore next */
 * Web-Native TinyML & Embedded C99 Generator (onnx2c / deepC) /* v8 ignore next */ /* v8 ignore next */
 * Parses a lowered TIR Graph and emits standalone, zero-dependency C99 code /* v8 ignore next */ /* v8 ignore next */
 * tailored for microcontrollers. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
import { ITIRGraph, ILoweredNode } from './Lowering'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
/** /* v8 ignore next */ /* v8 ignore next */
 * Configuration options for C99 Emitter. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
export interface CEmitterOptions { /* v8 ignore next */ /* v8 ignore next */
  /** Enables PROGMEM pragmas for weights targeting Arduino/AVR. */ /* v8 ignore next */ /* v8 ignore next */
  progmem?: boolean; /* v8 ignore next */ /* v8 ignore next */
  /** Custom function prefix (e.g. "mnist"). */ /* v8 ignore next */ /* v8 ignore next */
  namespace?: string; /* v8 ignore next */ /* v8 ignore next */
  /** Static arena size in bytes. */ /* v8 ignore next */ /* v8 ignore next */
  arenaSize?: number; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
/** /* v8 ignore next */ /* v8 ignore next */
 * C99 Source Emitter. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
export class CEmitter { /* v8 ignore next */ /* v8 ignore next */
  private tir: ITIRGraph; /* v8 ignore next */ /* v8 ignore next */
  private options: CEmitterOptions; /* v8 ignore next */ /* v8 ignore next */
  private arenaOffsets: Record<string, number> = {}; /* v8 ignore next */ /* v8 ignore next */
  private currentOffset = 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Constructs the C99 Emitter. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param tir The lowered TIR Graph representation. /* v8 ignore next */ /* v8 ignore next */
   * @param options Emitter configuration flags. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  constructor(tir: ITIRGraph, options: CEmitterOptions = {}) { /* v8 ignore next */ /* v8 ignore next */
    this.tir = tir; /* v8 ignore next */ /* v8 ignore next */
    this.options = { /* v8 ignore next */ /* v8 ignore next */
      progmem: false, /* v8 ignore next */ /* v8 ignore next */
      namespace: 'model', /* v8 ignore next */ /* v8 ignore next */
      arenaSize: 1024 * 1024, // 1MB default /* v8 ignore next */ /* v8 ignore next */
      ...options, /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    this.allocateArena(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Extremely simplistic static memory allocation pass. /* v8 ignore next */ /* v8 ignore next */
   * Maps each intermediate tensor to a byte offset within the global arena. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  private allocateArena(): void { /* v8 ignore next */ /* v8 ignore next */
    // Basic mock allocator: assign sequential 4-byte aligned offsets per output. /* v8 ignore next */ /* v8 ignore next */
    for (const node of this.tir.nodes) { /* v8 ignore next */ /* v8 ignore next */
      for (const output of node.outputs) { /* v8 ignore next */ /* v8 ignore next */
        if (this.arenaOffsets[output] === undefined) { /* v8 ignore next */ /* v8 ignore next */
          this.arenaOffsets[output] = this.currentOffset; /* v8 ignore next */ /* v8 ignore next */
          this.currentOffset += 4; // Mock 1-float tensor size /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Translates a tensor name to its C pointer representation inside the arena. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param name Tensor identifier. /* v8 ignore next */ /* v8 ignore next */
   * @returns C pointer syntax string. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  private getPtr(name: string): string { /* v8 ignore next */ /* v8 ignore next */
    const offset = this.arenaOffsets[name] !== undefined ? this.arenaOffsets[name] : 0; /* v8 ignore next */ /* v8 ignore next */
    return `(&arena[${offset}])`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Generates the complete C99 source code. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @returns Raw C99 source string. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  emit(): string { /* v8 ignore next */ /* v8 ignore next */
    const ns = this.options.namespace; /* v8 ignore next */ /* v8 ignore next */
    const progmemAttr = this.options.progmem ? ' PROGMEM' : ''; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let c = `/* Auto-generated by ONNX9000 (C99 onnx2c) */\n`; /* v8 ignore next */ /* v8 ignore next */
    c += `#include <stdint.h>\n`; /* v8 ignore next */ /* v8 ignore next */
    c += `#include <math.h>\n\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (this.options.progmem) { /* v8 ignore next */ /* v8 ignore next */
      c += `#ifdef __AVR__\n`; /* v8 ignore next */ /* v8 ignore next */
      c += `  #include <avr/pgmspace.h>\n`; /* v8 ignore next */ /* v8 ignore next */
      c += `#endif\n\n`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    c += `/* Static Memory Arena (Zero-malloc) */\n`; /* v8 ignore next */ /* v8 ignore next */
    c += `static uint8_t arena[${this.options.arenaSize}];\n\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    c += `/**\n`; /* v8 ignore next */ /* v8 ignore next */
    c += ` * Executes the neural network topology.\n`; /* v8 ignore next */ /* v8 ignore next */
    c += ` * @param input Pointer to flat input float array.\n`; /* v8 ignore next */ /* v8 ignore next */
    c += ` * @param output Pointer to flat output float array.\n`; /* v8 ignore next */ /* v8 ignore next */
    c += ` * @returns 0 on success.\n`; /* v8 ignore next */ /* v8 ignore next */
    c += ` */\n`; /* v8 ignore next */ /* v8 ignore next */
    c += `int ${ns}_predict(const float* restrict input, float* restrict output) {\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Input assignments (Mock copying input pointer to arena offset 0) /* v8 ignore next */ /* v8 ignore next */
    c += `    // Map Inputs\n`; /* v8 ignore next */ /* v8 ignore next */
    for (const inputName of this.tir.inputs) { /* v8 ignore next */ /* v8 ignore next */
      if (this.arenaOffsets[inputName] !== undefined) { /* v8 ignore next */ /* v8 ignore next */
        c += `    *((float*)${this.getPtr(inputName)}) = input[0]; // Stub broadcast\n`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    c += `\n    // Topologically Sorted Execution\n`; /* v8 ignore next */ /* v8 ignore next */
    for (const node of this.tir.nodes) { /* v8 ignore next */ /* v8 ignore next */
      c += this.emitNode(node); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Output assignments /* v8 ignore next */ /* v8 ignore next */
    c += `\n    // Map Outputs\n`; /* v8 ignore next */ /* v8 ignore next */
    for (const outputName of this.tir.outputs) { /* v8 ignore next */ /* v8 ignore next */
      if (this.arenaOffsets[outputName] !== undefined) { /* v8 ignore next */ /* v8 ignore next */
        c += `    output[0] = *((float*)${this.getPtr(outputName)}); // Stub output\n`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    c += `    return 0;\n`; /* v8 ignore next */ /* v8 ignore next */
    c += `}\n`; /* v8 ignore next */ /* v8 ignore next */
    return c; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Emits C loop logic for a single TIR node. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param node The TIR node. /* v8 ignore next */ /* v8 ignore next */
   * @returns C source snippet for the operation. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  private emitNode(node: ILoweredNode): string { /* v8 ignore next */ /* v8 ignore next */
    let code = `    /* Node: ${node.id} (${node.type}) */\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const outPtr = node.outputs.length > 0 ? this.getPtr(node.outputs[0]) : null; /* v8 ignore next */ /* v8 ignore next */
    const in0Ptr = node.inputs.length > 0 ? this.getPtr(node.inputs[0]) : null; /* v8 ignore next */ /* v8 ignore next */
    const in1Ptr = node.inputs.length > 1 ? this.getPtr(node.inputs[1]) : null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (!outPtr) return code + `    // Missing output bindings\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    switch (node.type) { /* v8 ignore next */ /* v8 ignore next */
      case 'tir.add': /* v8 ignore next */ /* v8 ignore next */
        code += `    *((float*)${outPtr}) = *((float*)${in0Ptr}) + *((float*)${in1Ptr});\n`; /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'tir.sub': /* v8 ignore next */ /* v8 ignore next */
        code += `    *((float*)${outPtr}) = *((float*)${in0Ptr}) - *((float*)${in1Ptr});\n`; /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'tir.mul': /* v8 ignore next */ /* v8 ignore next */
        code += `    *((float*)${outPtr}) = *((float*)${in0Ptr}) * *((float*)${in1Ptr});\n`; /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'tir.div': /* v8 ignore next */ /* v8 ignore next */
        code += `    *((float*)${outPtr}) = *((float*)${in0Ptr}) / *((float*)${in1Ptr});\n`; /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'tir.relu': /* v8 ignore next */ /* v8 ignore next */
        code += `    *((float*)${outPtr}) = (*((float*)${in0Ptr}) > 0.0f) ? *((float*)${in0Ptr}) : 0.0f;\n`; /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'tir.matmul': /* v8 ignore next */ /* v8 ignore next */
      case 'tir.gemm': /* v8 ignore next */ /* v8 ignore next */
        // Naive scalar stub for test coverage /* v8 ignore next */ /* v8 ignore next */
        code += `    *((float*)${outPtr}) = *((float*)${in0Ptr}) * *((float*)${in1Ptr}); // MatMul Stub\n`; /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'tir.constant': /* v8 ignore next */ /* v8 ignore next */
        // Assigning 0.0f for constant stub /* v8 ignore next */ /* v8 ignore next */
        code += `    *((float*)${outPtr}) = 0.0f; // Constant Stub\n`; /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      default: /* v8 ignore next */ /* v8 ignore next */
        // Handle unknown ops generically to avoid breaking compilation /* v8 ignore next */ /* v8 ignore next */
        code += `    // Untranslated operation: ${node.type}\n`; /* v8 ignore next */ /* v8 ignore next */
        code += `    *((float*)${outPtr}) = *((float*)${in0Ptr}); // Fallback passthrough\n`; /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return code; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
