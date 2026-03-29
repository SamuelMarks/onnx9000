/* eslint-disable */
// @ts-nocheck
export interface PipelineState {
  sourceFramework: string | null;
  targetFramework: string | null;
  activeFile: string | null;
  outputArtifacts?: object[];
}

/**
 * Represents a single step in the transformation pipeline.
 * Keeps track of the state before and after an operation to allow undo/redo.
 */
export class PipelineNode {
  public id: string;
  public timestamp: number;
  public state: PipelineState;
  public description: string;

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
