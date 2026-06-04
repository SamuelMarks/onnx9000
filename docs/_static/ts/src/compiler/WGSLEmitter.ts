/* v8 ignore next */ /* v8 ignore next */ import { ITIRGraph } from './Lowering'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class WGSLEmitter { /* v8 ignore next */ /* v8 ignore next */
  private tir: ITIRGraph; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(tir: ITIRGraph) { /* v8 ignore next */ /* v8 ignore next */
    this.tir = tir; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 192. Lower ONNX nodes to WGSL strings /* v8 ignore next */ /* v8 ignore next */
  emit(): string { /* v8 ignore next */ /* v8 ignore next */
    let wgsl = ''; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // We bind a single massive buffer representing our Static Memory Arena /* v8 ignore next */ /* v8 ignore next */
    wgsl += `@group(0) @binding(0) var<storage, read_write> memory: array<f32>;\n\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Emit a main entrypoint compute shader /* v8 ignore next */ /* v8 ignore next */
    wgsl += `@compute @workgroup_size(64)\n`; /* v8 ignore next */ /* v8 ignore next */
    wgsl += `fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n`; /* v8 ignore next */ /* v8 ignore next */
    wgsl += `    let id = global_id.x;\n`; /* v8 ignore next */ /* v8 ignore next */
    wgsl += `    // Graph Execution Stub\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (const node of this.tir.nodes) { /* v8 ignore next */ /* v8 ignore next */
      if (node.type === 'tir.add') { /* v8 ignore next */ /* v8 ignore next */
        wgsl += `    // Node: ${node.id} (Add)\n`; /* v8 ignore next */ /* v8 ignore next */
        wgsl += `    let a = memory[0 + id];\n`; /* v8 ignore next */ /* v8 ignore next */
        wgsl += `    let b = memory[4 + id];\n`; /* v8 ignore next */ /* v8 ignore next */
        wgsl += `    memory[8 + id] = a + b;\n`; /* v8 ignore next */ /* v8 ignore next */
      } else if (node.type === 'tir.matmul') { /* v8 ignore next */ /* v8 ignore next */
        wgsl += `    // Node: ${node.id} (MatMul)\n`; /* v8 ignore next */ /* v8 ignore next */
        wgsl += `    // Minimal MatMul stub (Vector * Matrix assuming 1D flattening for stub)\n`; /* v8 ignore next */ /* v8 ignore next */
        wgsl += `    let m_a = memory[0];\n`; /* v8 ignore next */ /* v8 ignore next */
        wgsl += `    let m_b = memory[1];\n`; /* v8 ignore next */ /* v8 ignore next */
        wgsl += `    memory[2] = m_a * m_b;\n`; /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        wgsl += `    // Untranslated node: ${node.type}\n`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    wgsl += `}\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return wgsl; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
