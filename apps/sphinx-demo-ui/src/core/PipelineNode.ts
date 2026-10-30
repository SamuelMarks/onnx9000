/**
 * @fileoverview PipelineNode.ts
 * Provides PipelineNode functionality for the Sphinx Demo UI.
 */
export interface PipelineState {
  /** Property `sourceFramework`. */
  sourceFramework: string | null;
  /** Property `targetFramework`. */
  targetFramework: string | null;
  /** Property `activeFile`. */
  activeFile: string | null;
  /** Property `outputArtifacts`. */
  outputArtifacts?: object[];
}

/**
 * Represents a single step in the transformation pipeline.
 * Keeps track of the state before and after an operation to allow undo/redo.
 */
export class PipelineNode {
  /** Property `id` of `PipelineNode`. */
  public id: string;
  /** Property `timestamp` of `PipelineNode`. */
  public timestamp: number;
  /** Property `state` of `PipelineNode`. */
  public state: PipelineState;
  /** Property `description` of `PipelineNode`. */
  public description: string;

  /**
   * Initializes a new `PipelineNode` instance.
   * @param state Parameter `state`.
   * @param description Parameter `description`.
   */
  constructor(state: PipelineState, description: string) {
    this.id =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2, 15);
    this.timestamp = Date.now();
    // Deep clone state to ensure immutability
    this.state = JSON.parse(JSON.stringify(state));
    this.description = description;
  }
}
