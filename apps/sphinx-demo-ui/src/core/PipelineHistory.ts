/* eslint-disable */
// @ts-nocheck
import { PipelineNode, PipelineState } from './PipelineNode';
import { globalEventBus } from './EventBus';

/**
 * Manages the state history of pipeline transitions.
 * Implements undo/redo mechanics.
 */
export class PipelineHistory {
  private history: PipelineNode[] = [];
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

  public canUndo(): boolean {
    return this.currentIndex > 0;
  }

  public canRedo(): boolean {
    return this.currentIndex < this.history.length - 1;
  }

  public getHistory(): PipelineNode[] {
    return this.history.slice(0, this.currentIndex + 1);
  }

  public clear(): void {
    this.history = [];
    this.currentIndex = -1;
  }
}
