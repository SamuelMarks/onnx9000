import { describe, it, expect, vi } from 'vitest';
import * as fetcher from '../src/parser/fetcher';

describe('index.ts deeper UI coverage', () => {
  it('should run everything in index', async () => {
    window.location.hash = 'url=https://github.com/test';

    // We mock fetchAndParseModel to return a specific graph to hit all renderSidebar branches
    vi.spyOn(fetcher, 'fetchAndParseModel').mockResolvedValue({
      inputs: [{ name: 'X', dtype: 'float32', shape: [1] }],
      outputs: [{ name: 'Y', dtype: 'float32', shape: [1] }],
      initializers: ['W'],
      nodes: [
        {
          name: 'AddNode',
          opType: 'Add',
          inputs: ['X'],
          outputs: ['Y'],
          attributes: {
            a: { type: 'FLOAT', value: 1.0 },
            b: { type: 'TENSOR', value: { formatData: () => 'val' } },
            c: { type: 'INTS', value: [1] },
            d: { type: 'STRINGS', value: ['a'] },
            e: { type: 'FLOATS', value: [1.0] },
          },
          domain: 'ai.onnx',
        },
      ],
      tensors: {
        W: { name: 'W', dtype: 'float32', shape: [1], size: 1, formatData: () => 'data' },
      },
      opsetImports: { '': 14, 'ai.onnx.ml': 2 },
    } as any);

    // Load index
    await import('../src/index');

    const searchBox = document.getElementById('search-box') as HTMLInputElement;
    const fileUpload = document.getElementById('file-upload') as HTMLInputElement;
    const canvas = document.getElementById('view') as HTMLCanvasElement;
    const searchResults = document.getElementById('search-results') as HTMLDivElement;
    const sidebar = document.getElementById('sidebar') as HTMLDivElement;

    // Simulate drop
    const dragover = new CustomEvent('dragover');
    (dragover as any).dataTransfer = { files: [] };
    window.dispatchEvent(dragover);

    const drop = new CustomEvent('drop');
    (drop as any).dataTransfer = { files: [new File([''], 'model.onnx')] };
    window.dispatchEvent(drop);

    // Wait for worker mock
    await new Promise((r) => setTimeout(r, 50));

    // Test the event handlers on search
    searchBox.value = 'Add';
    searchBox.dispatchEvent(new Event('input'));

    const children = searchResults.querySelectorAll('div');
    if (children.length > 0) {
      const child = children[0] as HTMLDivElement;
      child.dispatchEvent(new MouseEvent('mouseenter'));
      child.dispatchEvent(new MouseEvent('mouseleave'));

      // click
      child.dispatchEvent(new MouseEvent('click'));
      // this triggers renderSidebar
    }

    // Now test all renderSidebar branches by dispatching more clicks
    // renderer.onSelect was bound to renderSidebar. We can get it from the canvas mock or just simulate it.
    // wait, we can just trigger it using search!

    // Test Input
    searchBox.value = 'X';
    searchBox.dispatchEvent(new Event('input'));
    searchResults.querySelectorAll('div')[0]?.dispatchEvent(new MouseEvent('click'));

    // Test Output
    searchBox.value = 'Y';
    searchBox.dispatchEvent(new Event('input'));
    searchResults.querySelectorAll('div')[0]?.dispatchEvent(new MouseEvent('click'));

    // Test Constant
    searchBox.value = 'W';
    searchBox.dispatchEvent(new Event('input'));
    searchResults.querySelectorAll('div')[0]?.dispatchEvent(new MouseEvent('click'));

    // Test Missing
    searchBox.value = 'Missing';
    searchBox.dispatchEvent(new Event('input'));

    // Clear
    searchBox.value = '';
    searchBox.dispatchEvent(new Event('input'));

    // Trigger onSelect(null)
    const escEvent = new KeyboardEvent('keydown', { key: 'Escape' });
    window.dispatchEvent(escEvent);
  });
});
