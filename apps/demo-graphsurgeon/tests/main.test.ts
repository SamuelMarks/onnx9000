import { describe, it, expect, vi } from 'vitest';
import { initGraphSurgeonDemo } from '../src/main.js';

vi.mock('@onnx9000/modifier', () => ({
  GraphMutator: class {
    deleteNode = vi.fn();
  },
}));

describe('demo-graphsurgeon', () => {
  it('should mutate graph', async () => {
    document.body.innerHTML = '<button id="mutate-btn"></button><div id="surgeon-output"></div>';
    initGraphSurgeonDemo();
    document.getElementById('mutate-btn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('surgeon-output')?.innerText).toContain(
      'Success! Graph structure modified.',
    );
  });
});
