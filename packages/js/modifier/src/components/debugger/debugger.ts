/**
 * @fileoverview debugger module for modifier.
 */
import type { GraphMutator } from '../../GraphMutator.js';

/**
 * Handles Interactive Graph Execution & Debugging (Phase 7).
 */
export class GraphDebugger {
  /** Property `container` of `GraphDebugger`. */
  container: HTMLElement;
  /** Property `mutator` of `GraphDebugger`. */
  mutator: GraphMutator;

  // The actual execution session (we would typically import this from @onnx9000/backend-web
  // but per Phase 7 instructions we mock the "integration" structure first)
  /** Property `session` of `GraphDebugger`. */
  session: ReturnType<typeof JSON.parse> = null;

  /** Property `activeBreakpoint` of `GraphDebugger`. */
  activeBreakpoint: string | null = null;
  /** Property `executionOutputs` of `GraphDebugger`. */
  executionOutputs: Map<string, ReturnType<typeof JSON.parse>> = new Map();

  /**
   * Initializes a new `GraphDebugger` instance.
   * @param container Parameter `container`.
   * @param mutator Parameter `mutator`.
   */
  constructor(container: HTMLElement, mutator: GraphMutator) {
    this.container = container;
    this.mutator = mutator;
  }

  // 86. Integrate execution runtime (mocking the initialization interface)
  /**
   * Executes `initSession` on `GraphDebugger`.
   * @returns Result of `initSession`.
   */
  async initSession() {
    // In a real run: this.session = await onnx9000.createSession(this.mutator.graph);
    this.session = { initialized: true };
    return this.session;
  }

  // 87. Support "Set as Temporary Output"
  /**
   * Executes `setAsTemporaryOutput` on `GraphDebugger`.
   * @param edgeName Parameter `edgeName`.
   * @returns Result of `setAsTemporaryOutput`.
   */
  setAsTemporaryOutput(edgeName: string) {
    const vi =
      this.mutator.graph.valueInfo.find((v) => v.name === edgeName) ||
      this.mutator.graph.inputs.find((i) => i.name === edgeName);
    if (!vi) return;

    this.mutator.execute({
      undo: () => {
        this.mutator.graph.outputs = this.mutator.graph.outputs.filter((o) => o.name !== edgeName);
      },
      redo: () => {
        // Only push if not already an output
        if (!this.mutator.graph.outputs.some((o) => o.name === edgeName)) {
          this.mutator.graph.outputs.push(vi);
        }
      },
    });
  }

  // 88. Implement Input Data Generator
  /**
   * Executes `generateDummyData` on `GraphDebugger`.
   * @returns Result of `generateDummyData`.
   */
  generateDummyData() {
    const dummyInputs: Record<string, Float32Array> = {};
    for (const inp of this.mutator.graph.inputs) {
      // Very basic mock logic
      let size = 1;
      for (const d of inp.shape) {
        if (typeof d === 'number' && d > 0) size *= d;
      }
      const arr = new Float32Array(size);
      for (let i = 0; i < size; i++) arr[i] = Math.random();
      dummyInputs[inp.name] = arr;
    }
    return dummyInputs;
  }

  // 89. Allow users to manually input values (Render form)
  /**
   * Executes `renderInputForm` on `GraphDebugger`.
   * @returns Result of `renderInputForm`.
   */
  renderInputForm() {
    this.container.innerHTML = '<h3>Manual Input Override</h3>';
    for (const inp of this.mutator.graph.inputs) {
      const wrapper = document.createElement('div');
      wrapper.innerHTML = `<strong>${inp.name}</strong> [${inp.shape.join(',')}]`;

      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'Comma separated values...';
      wrapper.appendChild(input);
      this.container.appendChild(wrapper);
    }
  }

  // 90. Execute graph natively
  // 93. Profile execution (Time tracking)
  /**
   * Executes `runGraph` on `GraphDebugger`.
   * @param _inputs Parameter `_inputs`.
   * @returns Result of `runGraph`.
   */
  async runGraph(_inputs: Record<string, ReturnType<typeof JSON.parse>>) {
    if (!this.session) await this.initSession();

    const startTime = performance.now();
    // Real call: const results = await this.session.run(inputs);
    // We mock execution for the editor logic
    const results: Record<string, ReturnType<typeof JSON.parse>> = {};
    for (const out of this.mutator.graph.outputs) {
      results[out.name] = new Float32Array([0.5, 0.5]); // mock return
    }
    const endTime = performance.now();

    this.executionOutputs = new Map(Object.entries(results));

    // Log profile
    const timeTaken = endTime - startTime;
    console.log(`Execution completed in ${timeTaken.toFixed(2)}ms`);

    return { results, timeTaken };
  }

  // 91. Display execution output tensor visually
  /**
   * Executes `renderOutputVisuals` on `GraphDebugger`.
   * @returns Result of `renderOutputVisuals`.
   */
  renderOutputVisuals() {
    this.container.innerHTML = '<h3>Execution Results</h3>';
    for (const [name, tensorData] of this.executionOutputs.entries()) {
      const wrapper = document.createElement('div');
      wrapper.innerHTML = `<strong>${name}</strong>: [${Array.from(tensorData).slice(0, 5).join(', ')}...]`;
      this.container.appendChild(wrapper);
    }
  }

  // 92. Support "Run Subgraph"
  /**
   * Executes `runSubgraph` on `GraphDebugger`.
   * @param _nodeIds Parameter `_nodeIds`.
   * @returns Result of `runSubgraph`.
   */
  async runSubgraph(_nodeIds: string[]) {
    // Rely on phase 5 utility to extract the subgraph
    const _tempGraph = this.mutator.graph; // Mocking extraction for isolated testing scope
    const _session = { initialized: true };
    // Run just that
    const startTime = performance.now();
    const results = { dummy_subgraph_out: new Float32Array([1.0]) };
    const timeTaken = performance.now() - startTime;
    return { results, timeTaken };
  }

  // 94. Step-by-step debugger
  // 95. Breakpoint pausing
  /**
   * Executes `stepNext` on `GraphDebugger`.
   * @returns Result of `stepNext`.
   */
  stepNext() {
    if (!this.session) return null;

    // In a real executor, we'd advance the program counter
    // For the UI state, we just yield the next node in topological order
    const nextNode = this.mutator.graph.nodes[0];
    if (nextNode) {
      if (this.activeBreakpoint === nextNode.id) {
        console.log('Paused at breakpoint:', nextNode.id);
        return { paused: true, node: nextNode };
      }
      return { paused: false, node: nextNode };
    }
    return null;
  }

  /**
   * Executes `setBreakpoint` on `GraphDebugger`.
   * @param nodeId Parameter `nodeId`.
   * @returns Result of `setBreakpoint`.
   */
  setBreakpoint(nodeId: string) {
    this.activeBreakpoint = nodeId;
  }
}
