/**
 * @fileoverview PipelineHistory module for sphinx-demo-ui.
 */
import { globalEventBus } from './EventBus';
import { PipelineNode, type PipelineState } from './PipelineNode';

/**
 * Manages the state history of pipeline transitions.
 * Implements undo/redo mechanics.
 */
export class PipelineHistory {
  /** Property `history` of `PipelineHistory`. */
  private history: PipelineNode[] = [];
  /** Property `currentIndex` of `PipelineHistory`. */
  private currentIndex: number = -1;

  /**
   * Pushes a new state to the history. Truncates any forward-paths if we had previously undone.
   */
  public push(state: PipelineState, description: string): void {
    if (this.currentIndex < this.history.length - 1) {
      this.history = this.history.slice(0, this.currentIndex + 1);
    }

    const node = new PipelineNode(state, description);
    this.history.push(node);
    this.currentIndex = this.history.length - 1;

    globalEventBus.emit('PIPELINE_STEP_ADDED', node);
  }

  /**
   * Reverts to the previous state.
   */
  public undo(): PipelineState | null {
    if (this.canUndo()) {
      const removedNode = this.history[this.currentIndex];
      this.currentIndex--;
      const prevState = this.history[this.currentIndex].state;

      globalEventBus.emit('PIPELINE_STEP_REMOVED', removedNode);
      return JSON.parse(JSON.stringify(prevState));
    }
    return null;
  }

  /**
   * Re-applies a previously undone state.
   */
  public redo(): PipelineState | null {
    if (this.canRedo()) {
      this.currentIndex++;
      const nextNode = this.history[this.currentIndex];
      const nextState = nextNode.state;

      globalEventBus.emit('PIPELINE_STEP_ADDED', nextNode);
      return JSON.parse(JSON.stringify(nextState));
    }
    return null;
  }

  /**
   * Executes `canUndo` on `PipelineHistory`.
   * @returns Result of `canUndo`.
   */
  public canUndo(): boolean {
    return this.currentIndex > 0;
  }

  /**
   * Executes `canRedo` on `PipelineHistory`.
   * @returns Result of `canRedo`.
   */
  public canRedo(): boolean {
    return this.currentIndex < this.history.length - 1;
  }

  /**
   * Executes `getHistory` on `PipelineHistory`.
   * @returns Result of `getHistory`.
   */
  public getHistory(): PipelineNode[] {
    return this.history.slice(0, this.currentIndex + 1);
  }

  /** Executes `clear` on `PipelineHistory`. */
  public clear(): void {
    this.history = [];
    this.currentIndex = -1;
  }
}
