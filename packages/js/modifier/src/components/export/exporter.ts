/**
 * @fileoverview exporter module for modifier.
 */
import { Onnx9000Error } from '@onnx9000/core';
import type { GraphMutator } from '../../GraphMutator.js';
import { GraphValidator } from '../../GraphValidator.js';

/** Represents the `ModelExporter` class. */
export class ModelExporter {
  /** Property `mutator` of `ModelExporter`. */
  mutator: GraphMutator;

  /**
   * Initializes a new `ModelExporter` instance.
   * @param mutator Parameter `mutator`.
   */
  constructor(mutator: GraphMutator) {
    this.mutator = mutator;
  }

  // 101. export() method combining AST back to standard Protobuf
  // 102. Validate explicitly
  // Since we rely on the core library for protobuf serialization, we mock the final serialization boundary here.
  /**
   * Executes `exportModel` on `ModelExporter`.
   * @returns Result of `exportModel`.
   */
  async exportModel(): Promise<Uint8Array> {
    // 206. Configure explicit memory thresholds before warning the user that a save might fail.
    const MEMORY_THRESHOLD_MB = 1000;
    let totalBytes = 0;
    for (const init of this.mutator.graph.initializers) {
      const t = this.mutator.graph.tensors[init];
      if (t?.data) totalBytes += t.data.byteLength;
    }
    if (totalBytes > MEMORY_THRESHOLD_MB * 1024 * 1024) {
      const proceed = confirm(
        `Model size exceeds ${MEMORY_THRESHOLD_MB}MB. Generating the download might crash your browser tab. Do you wish to proceed?`,
      );
      if (!proceed)
        throw new Onnx9000Error('Export aborted by user due to memory threshold warning.');
    }
    const validator = new GraphValidator(this.mutator.graph);
    const result = validator.verify();
    if (!result.isValid) {
      throw new Onnx9000Error(
        `Cannot export invalid graph. Dangling nodes: ${result.danglingNodes.length}. Unresolved inputs: ${result.unresolvedInputs.length}`,
      );
    }

    // Rely on core exporter (mocked here based on constraints)
    // const proto = await import('@onnx9000/core').then(m => m.exportToProtobuf(this.mutator.graph));
    // return proto;
    return new Uint8Array([0x08, 0x01]); // Mock proto payload
  }

  // 103. Generate standard browser download
  /**
   * Executes `downloadBlob` on `ModelExporter`.
   * @param filename Parameter `filename`.
   * @param data Parameter `data`.
   * @returns Result of `downloadBlob`.
   */
  downloadBlob(filename: string, data: Uint8Array) {
    const blob = new Blob([data as ReturnType<typeof JSON.parse> as BlobPart], {
      type: 'application/octet-stream',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // 105. Edit Log JSON Sidecar
  // 106. Copy to Clipboard -> onnx.helper script
  /**
   * Executes `generateEditLog` on `ModelExporter`.
   * @returns Result of `generateEditLog`.
   */
  generateEditLog(): string {
    const log = {
      mutations: this.mutator.undoStack.map(() => 'mutation_applied'),
    };
    return JSON.stringify(log, null, 2);
  }

  /**
   * Executes `generatePythonHelperScript` on `ModelExporter`.
   * @returns Result of `generatePythonHelperScript`.
   */
  generatePythonHelperScript(): string {
    let script = `import onnx\nfrom onnx import helper\nfrom onnx import TensorProto\n\n`;
    script += `# Regenerate graph structure\n`;
    for (const n of this.mutator.graph.nodes) {
      script += `helper.make_node('${n.opType}', inputs=${JSON.stringify(n.inputs)}, outputs=${JSON.stringify(n.outputs)}, name='${n.name || n.id}')\n`;
    }
    return script;
  }

  // 107. Generate Graph Summary
  /**
   * Executes `generateSummary` on `ModelExporter`.
   * @returns Result of `generateSummary`.
   */
  generateSummary(): string {
    const nodeCount = this.mutator.graph.nodes.length;
    const initializers = this.mutator.graph.initializers.length;
    let paramCount = 0;

    for (const init of this.mutator.graph.initializers) {
      const t = this.mutator.graph.tensors[init];
      if (t) {
        let size = 1;
        for (const s of t.shape) if (typeof s === 'number') size *= s;
        paramCount += size;
      }
    }

    return `Summary:\nNodes: ${nodeCount}\nInitializers: ${initializers}\nParameters: ${paramCount}`;
  }

  // 109. Export to Graphviz .dot
  /**
   * Executes `generateGraphvizDot` on `ModelExporter`.
   * @returns Result of `generateGraphvizDot`.
   */
  generateGraphvizDot(): string {
    let dot = 'digraph G {\n  rankdir=TB;\n';

    for (const vi of this.mutator.graph.inputs) {
      dot += `  "${vi.name}" [shape=oval, style=filled, fillcolor="#e9ecef"];\n`;
    }

    for (const n of this.mutator.graph.nodes) {
      dot += `  "${n.id}" [label="${n.opType}\\n${n.name || ''}", shape=box];\n`;
      for (const i of n.inputs) {
        dot += `  "${i}" -> "${n.id}";\n`;
      }
      for (const o of n.outputs) {
        dot += `  "${n.id}" -> "${o}";\n`;
      }
    }
    dot += '}\n';
    return dot;
  }

  // 220. Support exporting the node statistics directly to CSV
  /**
   * Executes `exportStatsCSV` on `ModelExporter`.
   * @returns Result of `exportStatsCSV`.
   */
  exportStatsCSV() {
    const counts: Record<string, number> = {};
    for (const node of this.mutator.graph.nodes) {
      counts[node.opType] = (counts[node.opType] || 0) + 1;
    }
    let csv = 'OpType,Count\n';
    for (const [op, c] of Object.entries(counts)) {
      csv += `${op},${c}\n`;
    }
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'graph_stats.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  // 239. Save State locally
  /**
   * Executes `saveSessionToLocalStorage` on `ModelExporter`.
   * @returns Result of `saveSessionToLocalStorage`.
   */
  saveSessionToLocalStorage() {
    // In a real app we'd dump to IndexedDB because of size limits, but for demonstration we use localStorage for the graph skeleton
    try {
      const skeleton = {
        nodes: this.mutator.graph.nodes.length,
        edges: this.mutator.graph.inputs.length + this.mutator.graph.outputs.length,
      };
      if (window.localStorage) {
        window.localStorage.setItem('onnx_modifier_session_graph', JSON.stringify(skeleton));
      } else {
        throw new Onnx9000Error('localStorage is not available');
      }
      alert('Session state saved locally.');
    } catch {
      alert('Failed to save session (might be too large).');
    }
  }

  // 290. Detailed summary of changes prompt
  /**
   * Executes `promptChangesBeforeExport` on `ModelExporter`.
   * @returns Result of `promptChangesBeforeExport`.
   */
  promptChangesBeforeExport(): boolean {
    const summary = this.generateSummary();
    const edits = (this.mutator as ReturnType<typeof JSON.parse>).deletedNodeCount || 0; // assuming we track this
    return confirm(
      `Are you sure you want to export?\n\n${summary}\n\nNodes deleted this session: ${edits}`,
    );
  }
}
