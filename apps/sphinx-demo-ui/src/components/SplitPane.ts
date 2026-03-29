/* eslint-disable */
// @ts-nocheck
import { Component } from '../core/Component';

export type SplitOrientation = 'horizontal' | 'vertical';

export interface SplitPaneOptions {
  orientation: SplitOrientation;
  initialSplitRatio?: number; // 0.0 to 1.0 (default 0.5)
  minSize?: number;
  className?: string;
  storageKey?: string; // If provided, persists the ratio to localStorage
}

/**
 * A reusable, drag-to-resize split pane component.
 * Allows rendering two sub-components either side-by-side or stacked vertically.
 */
export class SplitPane extends Component<HTMLDivElement> {
  private pane1!: HTMLDivElement;
  private pane2!: HTMLDivElement;
  private divider!: HTMLDivElement;

  private isDragging = false;
  private currentRatio = 0.5;

  private options: SplitPaneOptions;
  private boundMouseMove!: (e: MouseEvent) => void;
  private boundMouseUp!: (e: MouseEvent) => void;

  constructor(options: SplitPaneOptions) {
    super();
    this.options = {
      initialSplitRatio: 0.5,
      minSize: 100,
      ...options
    };

    // Core element init must happen here now
    this.element = this.render();

    // Try to load from local storage
    if (this.options.storageKey) {
      const stored = localStorage.getItem(this.options.storageKey);
      if (stored) {
        const parsed = parseFloat(stored);
        if (!isNaN(parsed) && parsed > 0 && parsed < 1) {
          this.currentRatio = parsed;
        }
      } else {
        this.currentRatio = this.options.initialSplitRatio!;
      }
    } else {
      this.currentRatio = this.options.initialSplitRatio!;
    }

    this.applyRatio();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.width = '100%';
    container.style.height = '100%';
    container.style.flex = '1';
    container.style.overflow = 'hidden';
    if (this.options.className) container.className = this.options.className;

    if (this.options.orientation === 'vertical') {
      container.style.flexDirection = 'column';
    } else {
      container.style.flexDirection = 'row';
    }

    this.pane1 = document.createElement('div');
    this.pane1.style.flex = '1 1 0%';
    this.pane1.style.minWidth = '0';
    this.pane1.style.minHeight = '0';
    this.pane1.style.display = 'flex';
    this.pane1.style.flexDirection = 'column';

    this.pane2 = document.createElement('div');
    this.pane2.style.flex = '1 1 0%';
    this.pane2.style.minWidth = '0';
    this.pane2.style.minHeight = '0';
    this.pane2.style.display = 'flex';
    this.pane2.style.flexDirection = 'column';

    this.divider = document.createElement('div');

    // Accessibility for the divider
    this.divider.setAttribute('role', 'separator');
    this.divider.setAttribute('aria-orientation', this.options.orientation);
    this.divider.setAttribute('tabindex', '0');
    this.divider.className =
      this.options.orientation === 'vertical'
        ? 'demo-pane-divider-horizontal'
        : 'demo-pane-divider-vertical';
    this.divider.setAttribute('aria-label', `Resize ${this.options.orientation} pane`);

    if (this.options.orientation === 'vertical') {
    } else {
    }

    container.appendChild(this.pane1);
    container.appendChild(this.divider);
    container.appendChild(this.pane2);

    return container;
  }

  protected onMount(): void {
    // We bind DOM listeners to handle the drag lifecycle
    this.addDOMListener(this.divider, 'mousedown', (e) => this.onDragStart(e as MouseEvent));
    this.addDOMListener(this.divider, 'dblclick', () => this.resetRatio());
    this.addDOMListener(this.divider, 'keydown', (e) => this.onKeyDown(e as KeyboardEvent));

    this.boundMouseMove = this.onDrag.bind(this);
    this.boundMouseUp = this.onDragEnd.bind(this);

    // Add global window event listener cleanup
    this.onCleanup(() => {
      window.removeEventListener('mousemove', this.boundMouseMove);
      window.removeEventListener('mouseup', this.boundMouseUp);
    });
  }

  public getPanes(): { pane1: HTMLDivElement; pane2: HTMLDivElement } {
    return { pane1: this.pane1, pane2: this.pane2 };
  }

  private applyRatio(): void {
    const p1 = this.currentRatio * 100;
    const p2 = 100 - p1;
    this.pane1.style.flex = `0 0 calc(${p1}% - 8px)`;
    this.pane2.style.flex = `0 0 calc(${p2}% - 8px)`;
  }

  private resetRatio(): void {
    this.currentRatio = this.options.initialSplitRatio || 0.5;
    this.applyRatio();
    this.saveRatio();
  }

  private onKeyDown(e: KeyboardEvent): void {
    const step = 0.05; // 5% per key press
    let ratioChanged = false;

    if (this.options.orientation === 'horizontal') {
      if (e.key === 'ArrowLeft') {
        this.currentRatio = Math.max(0.1, this.currentRatio - step);
        ratioChanged = true;
      } else if (e.key === 'ArrowRight') {
        this.currentRatio = Math.min(0.9, this.currentRatio + step);
        ratioChanged = true;
      }
    } else {
      if (e.key === 'ArrowUp') {
        this.currentRatio = Math.max(0.1, this.currentRatio - step);
        ratioChanged = true;
      } else if (e.key === 'ArrowDown') {
        this.currentRatio = Math.min(0.9, this.currentRatio + step);
        ratioChanged = true;
      }
    }

    if (e.key === 'Home') {
      this.currentRatio = 0.1;
      ratioChanged = true;
    } else if (e.key === 'End') {
      this.currentRatio = 0.9;
      ratioChanged = true;
    } else if (e.key === 'Enter' || e.key === ' ') {
      this.resetRatio();
      ratioChanged = true;
    }

    if (ratioChanged) {
      e.preventDefault();
      this.applyRatio();
      this.saveRatio();
    }
  }

  private onDragStart(e: MouseEvent): void {
    e.preventDefault(); // Prevents text selection during drag
    this.isDragging = true;

    this.divider.classList.add('active');

    window.addEventListener('mousemove', this.boundMouseMove);
    window.addEventListener('mouseup', this.boundMouseUp);
  }

  private onDrag(e: MouseEvent): void {
    if (!this.isDragging) return;

    // Calculate new ratio
    const containerRect = this.element.getBoundingClientRect();

    let newRatio = this.currentRatio;

    if (this.options.orientation === 'horizontal') {
      // Horizontal split (left/right) -> Track Mouse X
      const minPixels = this.options.minSize || 0;
      let newWidth = e.clientX - containerRect.left;

      // Boundaries
      if (newWidth < minPixels) newWidth = minPixels;
      if (newWidth > containerRect.width - minPixels) newWidth = containerRect.width - minPixels;

      // If width is tiny, fallback safely
      if (containerRect.width > 0) {
        newRatio = newWidth / containerRect.width;
      }
    } else {
      // Vertical split (top/bottom) -> Track Mouse Y
      const minPixels = this.options.minSize || 0;
      let newHeight = e.clientY - containerRect.top;

      // Boundaries
      if (newHeight < minPixels) newHeight = minPixels;
      if (newHeight > containerRect.height - minPixels)
        newHeight = containerRect.height - minPixels;

      // If height is tiny, fallback safely
      if (containerRect.height > 0) {
        newRatio = newHeight / containerRect.height;
      }
    }

    this.currentRatio = newRatio;
    this.applyRatio();
  }

  private onDragEnd(): void {
    if (!this.isDragging) return;
    this.isDragging = false;

    window.removeEventListener('mousemove', this.boundMouseMove);
    window.removeEventListener('mouseup', this.boundMouseUp);

    this.saveRatio();

    // Dispatch a custom event from the element so tests or parent views can react
    this.element.dispatchEvent(
      new CustomEvent('split-resize', {
        detail: { ratio: this.currentRatio }
      })
    );
  }

  private saveRatio(): void {
    if (this.options.storageKey) {
      localStorage.setItem(this.options.storageKey, this.currentRatio.toString());
    }
  }
}
