import { describe, it, expect, vi } from 'vitest';
import { initNetronUI } from '../src/index.js';

// Mock worker since we are in node
(global as any).Worker = class {
  postMessage() {}
};
vi.mock('../src/render/canvas', () => ({
  CanvasRenderer: class {
    selectedNodes = [];
    setLayout() {}
    render() {}
    setFilterControlEdges() {}
    setCustomColorRegex() {}
    setSearchResults() {}
    focusNode() {}
  },
}));

describe('netron-ui index', () => {
  it('should inject UI', () => {
    initNetronUI();
    expect(document.getElementById('view')).toBeDefined();
    expect(document.getElementById('drop-zone')).toBeDefined();
  });
});
