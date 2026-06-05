// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { BottomContainer } from '../src/components/BottomContainer.js';
import { Logger } from '../src/core/Logger.js';

describe('BottomContainer', () => {
  it('should render', () => {
    Logger.getInstance();
    const container = new BottomContainer();
    document.body.appendChild(container.element);
    expect(container.element.className).toContain('demo-pane-bottom');
  });
});
