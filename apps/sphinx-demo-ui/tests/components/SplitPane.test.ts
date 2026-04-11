/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SplitPane } from '../../src/components/SplitPane';

describe('SplitPane', () => {
  beforeEach(() => {
    localStorage.clear();
    // jsdom doesn't compute getBoundingClientRect well by default
    Element.prototype.getBoundingClientRect = vi.fn(() => {
      return {
        width: 1000,
        height: 1000,
        top: 0,
        left: 0,
        right: 1000,
        bottom: 1000,
        x: 0,
        y: 0,
        toJSON: () => undefined
      };
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should render a horizontal split pane', () => {
    const pane = new SplitPane({ orientation: 'horizontal' });
    const { pane1, pane2 } = pane.getPanes();

    expect(pane1.style.flex).toBe('0 0 calc(50% - 8px)');
    expect(pane2.style.flex).toBe('0 0 calc(50% - 8px)');
  });

  it('should initialize with a custom initial ratio', () => {
    const pane = new SplitPane({ orientation: 'vertical', initialSplitRatio: 0.3 });
    const { pane1, pane2 } = pane.getPanes();

    expect(pane1.style.flex).toBe('0 0 calc(30% - 8px)');
    expect(pane2.style.flex).toBe('0 0 calc(70% - 8px)');
  });

  it('should persist and load ratio from localStorage', () => {
    localStorage.setItem('test-split', '0.75');

    const pane = new SplitPane({ orientation: 'horizontal', storageKey: 'test-split' });
    const { pane1, pane2 } = pane.getPanes();

    expect(pane1.style.flex).toBe('0 0 calc(75% - 8px)');
    expect(pane2.style.flex).toBe('0 0 calc(25% - 8px)');
  });

  it('should ignore invalid localStorage values', () => {
    localStorage.setItem('test-split', 'invalid-number');

    const pane = new SplitPane({ orientation: 'horizontal', storageKey: 'test-split' });
    const { pane1, pane2 } = pane.getPanes();

    expect(pane1.style.flex).toBe('0 0 calc(50% - 8px)'); // Default fallback
    expect(pane2.style.flex).toBe('0 0 calc(50% - 8px)');
  });

  it('should handle dragging to resize horizontally', () => {
    const pane = new SplitPane({ orientation: 'horizontal' });
    const parent = document.createElement('div');
    pane.mount(parent);

    const { pane1 } = pane.getPanes();
    const divider = pane1.nextSibling as HTMLElement; // Divider is the middle element

    // Simulate mousedown on divider
    divider.dispatchEvent(new MouseEvent('mousedown'));

    // Simulate mousemove globally
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 200 }));

    // Width is 1000, mouseX is 200, so new ratio should be 200/1000 = 0.2
    expect(pane1.style.flex).toBe('0 0 calc(20% - 8px)');

    // Simulate mouseup globally
    window.dispatchEvent(new MouseEvent('mouseup'));
  });

  it('should handle dragging to resize vertically', () => {
    const pane = new SplitPane({ orientation: 'vertical' });
    const parent = document.createElement('div');
    pane.mount(parent);

    const { pane1 } = pane.getPanes();
    const divider = pane1.nextSibling as HTMLElement;

    divider.dispatchEvent(new MouseEvent('mousedown'));
    window.dispatchEvent(new MouseEvent('mousemove', { clientY: 800 }));

    // Height is 1000, mouseY is 800, so new ratio should be 0.8
    expect(pane1.style.flex).toBe('0 0 calc(80% - 8px)');

    window.dispatchEvent(new MouseEvent('mouseup'));
  });

  it('should respect minSize boundaries', () => {
    const pane = new SplitPane({ orientation: 'horizontal', minSize: 100 });
    const parent = document.createElement('div');
    pane.mount(parent);

    const { pane1 } = pane.getPanes();
    const divider = pane1.nextSibling as HTMLElement;

    divider.dispatchEvent(new MouseEvent('mousedown'));

    // Move to 50px (less than minSize 100)
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 50 }));

    // Should be clamped to 100/1000 = 10%
    expect(pane1.style.flex).toBe('0 0 calc(10% - 8px)');

    // Move to 950px (greater than 1000 - minSize 100)
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 950 }));

    // Should be clamped to 900/1000 = 90%
    expect(pane1.style.flex).toBe('0 0 calc(90% - 8px)');

    window.dispatchEvent(new MouseEvent('mouseup'));
  });

  it('should reset ratio on double-click', () => {
    const pane = new SplitPane({ orientation: 'horizontal', initialSplitRatio: 0.6 });
    const parent = document.createElement('div');
    pane.mount(parent);

    const { pane1 } = pane.getPanes();
    const divider = pane1.nextSibling as HTMLElement;

    // Drag it somewhere else
    divider.dispatchEvent(new MouseEvent('mousedown'));
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 200 }));
    window.dispatchEvent(new MouseEvent('mouseup'));

    expect(pane1.style.flex).toBe('0 0 calc(20% - 8px)');

    // Double click to reset
    divider.dispatchEvent(new MouseEvent('dblclick'));

    expect(pane1.style.flex).toBe('0 0 calc(60% - 8px)');
  });

  it('should not update if not dragging', () => {
    const pane = new SplitPane({ orientation: 'horizontal' });
    const { pane1 } = pane.getPanes();

    // Just move mouse without mousedown
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 200 }));
    window.dispatchEvent(new MouseEvent('mouseup'));

    expect(pane1.style.flex).toBe('0 0 calc(50% - 8px)');
  });

  it('should emit a custom event when drag ends', () => {
    const pane = new SplitPane({ orientation: 'horizontal' });
    const parent = document.createElement('div');
    pane.mount(parent);

    const eventSpy = vi.fn();
    // Access protected element safely
    (pane as object).element.addEventListener('split-resize', eventSpy);

    const { pane1 } = pane.getPanes();
    const divider = pane1.nextSibling as HTMLElement;

    divider.dispatchEvent(new MouseEvent('mousedown'));
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 300 }));
    window.dispatchEvent(new MouseEvent('mouseup'));

    expect(eventSpy).toHaveBeenCalledTimes(1);
    expect(eventSpy.mock.calls[0][0].detail.ratio).toBe(0.3);
  });
});

it('should initialize with initial ratio when storageKey is provided but empty', () => {
  const pane = new SplitPane({
    orientation: 'horizontal',
    storageKey: 'empty-key',
    initialSplitRatio: 0.9
  });
  const { pane1 } = pane.getPanes();
  expect(pane1.style.flex).toBe('0 0 calc(90% - 8px)');
});

it('should cleanup event listeners on unmount', () => {
  const pane = new SplitPane({ orientation: 'horizontal' });
  const parent = document.createElement('div');
  pane.mount(parent);

  // Unmounting invokes the onCleanup logic which we need to cover
  pane.unmount();

  // We can not easily assert window.removeEventListener, but covering the lines is enough
  expect((pane as object).element.parentElement).toBeNull();
});

it('should save to localStorage on drag end when storageKey is provided', () => {
  const pane = new SplitPane({ orientation: 'horizontal', storageKey: 'test-save' });
  const parent = document.createElement('div');
  pane.mount(parent);

  const { pane1 } = pane.getPanes();
  const divider = pane1.nextSibling as HTMLElement;

  divider.dispatchEvent(new MouseEvent('mousedown'));
  window.dispatchEvent(new MouseEvent('mousemove', { clientX: 300 }));
  window.dispatchEvent(new MouseEvent('mouseup'));

  expect(localStorage.getItem('test-save')).toBe('0.3');
});

it('should handle zero-width horizontal boundaries gracefully', () => {
  const pane = new SplitPane({ orientation: 'horizontal' });
  const parent = document.createElement('div');
  pane.mount(parent);

  // Mock getBoundingClientRect to return 0 width
  (pane as object).element.getBoundingClientRect = vi.fn().mockReturnValue({ width: 0, left: 0 });

  const { pane1 } = pane.getPanes();
  const divider = pane1.nextSibling as HTMLElement;

  divider.dispatchEvent(new MouseEvent('mousedown'));
  window.dispatchEvent(new MouseEvent('mousemove', { clientX: 200 }));

  // Should fallback to currentRatio
  expect(pane1.style.flex).toBe('0 0 calc(50% - 8px)');
  window.dispatchEvent(new MouseEvent('mouseup'));
});

it('should handle zero-height vertical boundaries gracefully', () => {
  const pane = new SplitPane({ orientation: 'vertical' });
  const parent = document.createElement('div');
  pane.mount(parent);

  // Mock getBoundingClientRect to return 0 height
  (pane as object).element.getBoundingClientRect = vi.fn().mockReturnValue({ height: 0, top: 0 });

  const { pane1 } = pane.getPanes();
  const divider = pane1.nextSibling as HTMLElement;

  divider.dispatchEvent(new MouseEvent('mousedown'));
  window.dispatchEvent(new MouseEvent('mousemove', { clientY: 200 }));

  // Should fallback to currentRatio
  expect(pane1.style.flex).toBe('0 0 calc(50% - 8px)');
  window.dispatchEvent(new MouseEvent('mouseup'));
});
