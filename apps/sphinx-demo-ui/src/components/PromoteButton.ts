/* eslint-disable */
// @ts-nocheck
import { Component } from '../core/Component';
import { globalEventBus } from '../core/EventBus';

export class PromoteButton extends Component<HTMLButtonElement> {
  private isProcessing = false;

  constructor() {
    super();
    this.element = this.render();
  }

  protected render(): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.className = 'demo-btn-promote';
    btn.innerHTML = `&#8592; Promote to Source`;
    btn.title = 'Shift target artifact back to source to continue the pipeline';
    btn.disabled = true; // Disabled until a valid artifact exists
    return btn;
  }

  protected onMount(): void {
    this.addDOMListener(this.element, 'click', () => {
      if (!this.element.disabled) {
        globalEventBus.emit('PROMOTE_TARGET_TO_SOURCE');
      }
    });

    // Disable during WASM loading or processing
    this.onCleanup(
      globalEventBus.on<boolean>('WASM_PROCESSING_START', () => {
        this.isProcessing = true;
        this.element.disabled = true;
      })
    );

    this.onCleanup(
      globalEventBus.on<boolean>('WASM_PROCESSING_END', (success) => {
        this.isProcessing = false;
        this.element.disabled = !success;
      })
    );

    // Enable when an artifact is generated successfully
    this.onCleanup(
      globalEventBus.on<object>('TARGET_ARTIFACT_GENERATED', () => {
        if (!this.isProcessing) {
          this.element.disabled = false;
        }
      })
    );

    // Clear disables it
    this.onCleanup(
      globalEventBus.on<object>('TARGET_CLEARED', () => {
        this.element.disabled = true;
      })
    );
  }
}
