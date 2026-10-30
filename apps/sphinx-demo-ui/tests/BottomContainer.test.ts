import { describe, expect, it, vi } from 'vitest';
import { BottomContainer } from '../src/components/BottomContainer.js';
import { Logger } from '../src/core/Logger.js';

vi.mock('cytoscape', () => ({
  default: vi.fn().mockReturnValue({
    on: vi.fn(),
    destroy: vi.fn(),
    elements: () => ({ remove: vi.fn() }),
    add: vi.fn(),
    layout: () => ({ run: vi.fn() }),
    center: vi.fn(),
  }),
}));

describe('BottomContainer', () => {
  it('should render', () => {
    Logger.getInstance();
    const container = new BottomContainer();
    document.body.appendChild(container.element);
    expect(container.element.className).toContain('demo-pane-bottom');
  });
});
