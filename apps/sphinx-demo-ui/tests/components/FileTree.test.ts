import { describe, it, expect, vi } from 'vitest';
import { FileTree, FileNode } from '../../src/components/FileTree';

describe('FileTree', () => {
  const sampleData: FileNode = {
    name: 'src',
    type: 'directory',
    path: '/src',
    children: [
      { name: 'main.ts', type: 'file', path: '/src/main.ts' },
      {
        name: 'utils',
        type: 'directory',
        path: '/src/utils',
        children: [{ name: 'helper.ts', type: 'file', path: '/src/utils/helper.ts' }]
      }
    ]
  };

  it('should initialize and auto-expand root', () => {
    const tree = new FileTree({ root: sampleData });
    const el = (tree as any).element as HTMLElement;

    expect(el.className).toBe('demo-file-tree');

    const rootNode = el.querySelector('[data-path="/src"]') as HTMLElement;
    expect(rootNode.getAttribute('aria-expanded')).toBe('true');

    // Children of root should be visible
    const childrenContainer = rootNode.querySelector('.demo-file-tree-children') as HTMLElement;
    expect(childrenContainer.style.display).not.toBe('none');
  });

  it('should render file nodes properly', () => {
    const tree = new FileTree({ root: sampleData });
    const el = (tree as any).element as HTMLElement;

    const fileNode = el.querySelector('[data-path="/src/main.ts"]') as HTMLElement;
    expect(fileNode).not.toBeNull();
    expect(fileNode.textContent).toContain('📄');
    expect(fileNode.textContent).toContain('main.ts');
  });

  it('should handle expanding and collapsing directories', () => {
    const tree = new FileTree({ root: sampleData });
    const el = (tree as any).element as HTMLElement;
    tree.mount(document.body);

    const utilsNode = el.querySelector('[data-path="/src/utils"]') as HTMLElement;
    const utilsLabel = utilsNode.querySelector('.demo-file-tree-label') as HTMLElement;
    const utilsChildren = utilsNode.querySelector('.demo-file-tree-children') as HTMLElement;

    // Initially collapsed
    expect(utilsChildren.style.display).toBe('none');

    // Click to expand
    utilsLabel.click();
    expect((tree as any).expandedPaths.has('/src/utils')).toBe(true);

    // After re-render, we need to query again
    const updatedUtilsNode = el.querySelector('[data-path="/src/utils"]') as HTMLElement;
    const updatedChildren = updatedUtilsNode.querySelector(
      '.demo-file-tree-children'
    ) as HTMLElement;
    expect(updatedChildren.style.display).not.toBe('none');

    // Click to collapse
    const updatedLabel = updatedUtilsNode.querySelector('.demo-file-tree-label') as HTMLElement;
    updatedLabel.click();
    expect((tree as any).expandedPaths.has('/src/utils')).toBe(false);

    tree.unmount();
  });

  it('should handle selecting a file and trigger callback', () => {
    const onSelectSpy = vi.fn();
    const tree = new FileTree({ root: sampleData, onSelect: onSelectSpy });
    const el = (tree as any).element as HTMLElement;
    tree.mount(document.body);

    const fileNode = el.querySelector('[data-path="/src/main.ts"]') as HTMLElement;
    const fileLabel = fileNode.querySelector('.demo-file-tree-label') as HTMLElement;

    fileLabel.click();

    expect(onSelectSpy).toHaveBeenCalledWith('/src/main.ts');

    // Query again after re-render
    const updatedFileNode = el.querySelector('[data-path="/src/main.ts"]') as HTMLElement;
    expect(updatedFileNode.getAttribute('aria-selected')).toBe('true');
    expect(
      updatedFileNode.querySelector('.demo-file-tree-label')?.classList.contains('selected')
    ).toBe(true);

    tree.unmount();
  });

  it('should update data successfully', () => {
    const tree = new FileTree({ root: sampleData });

    const newData: FileNode = {
      name: 'tests',
      type: 'directory',
      path: '/tests',
      children: [{ name: 'app.test.ts', type: 'file', path: '/tests/app.test.ts' }]
    };

    tree.updateData(newData);

    const el = (tree as any).element as HTMLElement;
    expect(el.querySelector('[data-path="/tests"]')).not.toBeNull();
    expect(el.querySelector('[data-path="/src"]')).toBeNull();
  });

  it('should safely ignore clicks outside of valid labels', () => {
    const tree = new FileTree({ root: sampleData });
    const el = (tree as any).element as HTMLElement;
    tree.mount(document.body);

    // Clicking directly on the tree container shouldn't throw or do anything
    el.click();

    // Or clicking a nested ul
    const ul = el.querySelector('ul');
    if (ul) ul.click();

    expect((tree as any).selectedPath).toBeNull();

    tree.unmount();
  });

  it('should ignore clicks on nodes not found in data', () => {
    const tree = new FileTree({ root: sampleData });
    const el = (tree as any).element as HTMLElement;
    tree.mount(document.body);

    // Manipulate DOM to have a bad path
    const fileNode = el.querySelector('[data-path="/src/main.ts"]') as HTMLElement;
    fileNode.setAttribute('data-path', '/fake/path');

    const fileLabel = fileNode.querySelector('.demo-file-tree-label') as HTMLElement;
    fileLabel.click();

    // Nothing selected
    expect((tree as any).selectedPath).toBeNull();

    tree.unmount();
  });

  it('should handle broken DOM clicks gracefully', () => {
    const tree = new FileTree({ root: sampleData });
    const el = (tree as any).element as HTMLElement;
    tree.mount(document.body);

    const fakeLabel = document.createElement('div');
    fakeLabel.className = 'demo-file-tree-label';
    el.appendChild(fakeLabel);
    fakeLabel.click();

    const fakeNode = document.createElement('li');
    fakeNode.className = 'demo-file-tree-node';
    const fakeLabel2 = document.createElement('div');
    fakeLabel2.className = 'demo-file-tree-label';
    fakeNode.appendChild(fakeLabel2);
    el.appendChild(fakeNode);
    fakeLabel2.click();

    expect((tree as any).selectedPath).toBeNull();
  });
});
