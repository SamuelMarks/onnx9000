/**
 * @fileoverview SplitPane.ts
 * Provides SplitPane functionality for the Sphinx Demo UI.
 */
import { Component } from '../core/Component';

/** Type alias `SplitOrientation`. */
export type SplitOrientation = 'horizontal' | 'vertical';

/** Interface `SplitPaneOptions` definition. */
export interface SplitPaneOptions {
  /** Property `orientation`. */
  orientation: SplitOrientation;
  /** Property `initialSplitRatio`. */
  initialSplitRatio?: number; // 0.0 to 1.0 (default 0.5)
  /** Property `minSize`. */
  minSize?: number;
  /** Property `className`. */
  className?: string;
  /** Property `storageKey`. */
  storageKey?: string; // If provided, persists the ratio to localStorage
}

/**
 * A reusable, drag-to-resize split pane component.
 * Allows rendering two sub-components either side-by-side or stacked vertically.
 */
export class SplitPane extends Component<HTMLDivElement> {
  /** Property `pane1` of `SplitPane`. */
  private pane1: HTMLDivElement = document.createElement('div');
  /** Property `pane2` of `SplitPane`. */
  private pane2: HTMLDivElement = document.createElement('div');
  /** Property `divider` of `SplitPane`. */
  private divider: HTMLDivElement = document.createElement('div');

  /** Property `currentRatio` of `SplitPane`. */
  private currentRatio = 0.5;

  /** Property `options` of `SplitPane`. */
  private options: SplitPaneOptions;
  /** Property `boundMouseMove` of `SplitPane`. */
  private boundMouseMove: (e: MouseEvent) => void;
  /** Property `boundMouseUp` of `SplitPane`. */
  private boundMouseUp: () => void;

  /**
   * Initializes a new `SplitPane` instance.
   * @param options Parameter `options`.
   */
  constructor(options: SplitPaneOptions) {
    super();
    this.boundMouseMove = (e: MouseEvent) => this.onDrag(e);
    this.boundMouseUp = () => this.onDragEnd();
    this.options = {
      initialSplitRatio: 0.5,
      minSize: 100,
      ...options,
    };

    // Core element init must happen here now
    this.element = this.render();

    // Try to load from local storage
    if (this.options.storageKey) {
      const stored = localStorage.getItem(this.options.storageKey);
      if (stored) {
        const parsed = parseFloat(stored);
        if (!Number.isNaN(parsed) && parsed > 0 && parsed < 1) {
          this.currentRatio = parsed;
        }
      } else {
        this.currentRatio = this.options.initialSplitRatio as number;
      }
    } else {
      this.currentRatio = this.options.initialSplitRatio as number;
    }

    this.applyRatio();
  }

  /**
   * Executes `render` on `SplitPane`.
   * @returns Result of `render`.
   */
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

    container.appendChild(this.pane1);
    container.appendChild(this.divider);
    container.appendChild(this.pane2);

    return container;
  }

  /** Executes `onMount` on `SplitPane`. */
  protected onMount(): void {
    // We bind DOM listeners to handle the drag lifecycle
    this.addDOMListener(this.divider, 'mousedown', (e) => this.onDragStart(e as MouseEvent));
    this.addDOMListener(this.divider, 'dblclick', () => this.resetRatio());
    this.addDOMListener(this.divider, 'keydown', (e) => this.onKeyDown(e as KeyboardEvent));

    // Add global window event listener cleanup
    this.onCleanup(() => {
      window.removeEventListener('mousemove', this.boundMouseMove);
      window.removeEventListener('mouseup', this.boundMouseUp);
    });
  }

  /**
   * Executes `getPanes` on `SplitPane`.
   * @returns Result of `getPanes`.
   */
  public getPanes(): { pane1: HTMLDivElement; pane2: HTMLDivElement } {
    return { pane1: this.pane1, pane2: this.pane2 };
  }

  /** Executes `applyRatio` on `SplitPane`. */
  private applyRatio(): void {
    const p1 = this.currentRatio * 100;
    const p2 = 100 - p1;
    this.pane1.style.flex = `0 0 calc(${p1}% - 8px)`;
    this.pane2.style.flex = `0 0 calc(${p2}% - 8px)`;
  }

  /** Executes `resetRatio` on `SplitPane`. */
  private resetRatio(): void {
    this.currentRatio = this.options.initialSplitRatio as number;
    this.applyRatio();
    this.saveRatio();
  }

  /**
   * Executes `onKeyDown` on `SplitPane`.
   * @param e Parameter `e`.
   */
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

  /**
   * Executes `onDragStart` on `SplitPane`.
   * @param e Parameter `e`.
   */
  private onDragStart(e: MouseEvent): void {
    e.preventDefault(); // Prevents text selection during drag

    this.divider.classList.add('active');

    window.addEventListener('mousemove', this.boundMouseMove);
    window.addEventListener('mouseup', this.boundMouseUp);
  }

  /**
   * Executes `onDrag` on `SplitPane`.
   * @param e Parameter `e`.
   */
  private onDrag(e: MouseEvent): void {
    // Calculate new ratio
    const containerRect = this.element.getBoundingClientRect();

    let newRatio = this.currentRatio;

    if (this.options.orientation === 'horizontal') {
      // Horizontal split (left/right) -> Track Mouse X
      const minPixels = this.options.minSize as number;
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
      const minPixels = this.options.minSize as number;
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

  /** Executes `onDragEnd` on `SplitPane`. */
  private onDragEnd(): void {
    window.removeEventListener('mousemove', this.boundMouseMove);
    window.removeEventListener('mouseup', this.boundMouseUp);

    this.saveRatio();

    // Dispatch a custom event from the element so tests or parent views can react
    this.element.dispatchEvent(
      new CustomEvent('split-resize', {
        detail: { ratio: this.currentRatio },
      }),
    );
  }

  /** Executes `saveRatio` on `SplitPane`. */
  private saveRatio(): void {
    if (this.options.storageKey) {
      localStorage.setItem(this.options.storageKey, this.currentRatio.toString());
    }
  }
}
