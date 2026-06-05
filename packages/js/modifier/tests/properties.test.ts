import { describe, it, expect } from 'vitest';
import { PropertiesPanel } from '../src/components/properties.js';

describe('PropertiesPanel', () => {
  it('should render graph properties', () => {
    const container = document.createElement('div');
    const mutator: any = {};
    const panel = new PropertiesPanel(container, mutator);

    const g: any = {
      name: 'test',
      inputs: [],
      outputs: [],
      valueInfo: [],
      nodes: [],
      initializers: [],
      tensors: {},
      opsetImports: {},
    };
    panel.renderGraphProperties(g);

    expect(container.innerHTML).toContain('test');
  });
});
