// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { SplitPane } from '../src/components/SplitPane.js';

describe('SplitPane', () => {
  it('should render and resize via keyboard', () => {
    const pane = new SplitPane({ orientation: 'horizontal' });
    document.body.appendChild(pane.element);

    const divider = pane.element.querySelector('[role="separator"]') as HTMLElement;
    expect(divider).not.toBeNull();

    const event = new KeyboardEvent('keydown', { key: 'ArrowRight' });
    divider.dispatchEvent(event);

    const panes = pane.getPanes();
    expect(panes.pane1.style.flex).not.toBe('');
  });
});
