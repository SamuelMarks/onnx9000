import { Graph } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { PropertiesPanel } from '../src/components/properties.js';
import { GraphMutator } from '../src/GraphMutator.js';

describe('PropertiesPanel', () => {
  it('should render graph properties', () => {
    const container = document.createElement('div');
    const g = new Graph('test');
    const mutator = new GraphMutator(g);
    const panel = new PropertiesPanel(container, mutator);
    panel.renderGraphProperties(g);

    expect((container.querySelector('input') as HTMLInputElement).value).toBe('test');
  });
});
