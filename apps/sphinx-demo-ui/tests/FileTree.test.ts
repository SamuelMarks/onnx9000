// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { FileTree } from '../src/components/FileTree.js';

describe('FileTree', () => {
  it('should render and select', () => {
    let selected = '';
    const tree = new FileTree({
      root: {
        name: 'root',
        type: 'directory',
        path: '/',
        children: [{ name: 'file.txt', type: 'file', path: '/file.txt' }],
      },
      onSelect: (p) => {
        selected = p;
      },
    });
    tree.mount(document.body);

    const fileNode = tree.element.querySelector(
      '[data-path="/file.txt"] .demo-file-tree-label',
    ) as HTMLElement;
    fileNode?.click();

    expect(selected).toBe('/file.txt');
    expect(tree.getSelectedPath()).toBe('/file.txt');
  });
});
