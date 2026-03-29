/* eslint-disable */
// @ts-nocheck
import { Component } from '../core/Component';

export interface FileNode {
  name: string;
  type: 'file' | 'directory';
  path: string; // The full path
  content?: string; // Optional raw file content for mock data
  children?: FileNode[];
}

export interface FileTreeOptions {
  root: FileNode;
  onSelect?: (path: string) => void;
}

/**
 * A highly reusable, nested file tree component for the Demo UI.
 */
export class FileTree extends Component<HTMLDivElement> {
  private options: FileTreeOptions;
  private selectedPath: string | null = null;
  private expandedPaths: Set<string> = new Set();

  constructor(options: FileTreeOptions) {
    super();
    this.options = options;

    // Auto-expand root by default
    if (this.options.root.type === 'directory') {
      this.expandedPaths.add(this.options.root.path);
    }

    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-file-tree';
    container.setAttribute('role', 'tree');

    const rootUList = this.renderTree(this.options.root, true);
    container.appendChild(rootUList);

    return container;
  }

  private renderTree(node: FileNode, isRoot: boolean): HTMLUListElement {
    const ul = document.createElement('ul');
    if (!isRoot) {
      ul.className = 'demo-file-tree-nested';
    }

    // Treat root node differently if it's just a grouping wrapper, or render it.
    // For this demo, let's render the root as well.
    const li = document.createElement('li');
    li.className = 'demo-file-tree-node';
    li.setAttribute('role', 'treeitem');
    li.setAttribute('data-path', node.path);

    const labelDiv = document.createElement('div');
    labelDiv.className = 'demo-file-tree-label';

    const icon = document.createElement('span');
    icon.className = 'demo-file-tree-icon';

    if (node.type === 'directory') {
      const isExpanded = this.expandedPaths.has(node.path);
      icon.textContent = isExpanded ? '📂' : '📁';
      li.setAttribute('aria-expanded', isExpanded.toString());
      labelDiv.classList.add('demo-tree-dir');
    } else {
      icon.textContent = '📄';
      labelDiv.classList.add('demo-tree-file');
      if (node.path === this.selectedPath) {
        labelDiv.classList.add('selected');
        li.setAttribute('aria-selected', 'true');
      }
    }

    const text = document.createElement('span');
    text.textContent = node.name;

    labelDiv.appendChild(icon);
    labelDiv.appendChild(text);
    li.appendChild(labelDiv);

    if (node.type === 'directory' && node.children) {
      const childrenContainer = document.createElement('div');
      childrenContainer.className = 'demo-file-tree-children';

      if (!this.expandedPaths.has(node.path)) {
        childrenContainer.style.display = 'none';
      }

      for (const child of node.children) {
        const childUl = this.renderTree(child, false);
        childrenContainer.appendChild(childUl);
      }
      li.appendChild(childrenContainer);
    }

    ul.appendChild(li);
    return ul;
  }

  protected onMount(): void {
    this.addDOMListener(this.element, 'click', (e) => {
      const target = e.target as HTMLElement;
      const label = target.closest('.demo-file-tree-label');
      if (!label) return;

      const li = label.closest('.demo-file-tree-node');
      if (!li) return;

      const path = li.getAttribute('data-path');
      if (!path) return;

      // Find node in data
      const node = this.findNode(this.options.root, path);
      if (!node) return;

      if (node.type === 'directory') {
        this.toggleDirectory(node.path);
      } else {
        this.selectFile(node.path);
      }
    });
  }

  private toggleDirectory(path: string): void {
    if (this.expandedPaths.has(path)) {
      this.expandedPaths.delete(path);
    } else {
      this.expandedPaths.add(path);
    }
    this.reRender();
  }

  public getSelectedPath(): string | null {
    return this.selectedPath;
  }

  public selectFile(path: string): void {
    this.selectedPath = path;
    this.reRender();

    if (this.options.onSelect) {
      this.options.onSelect(path);
    }
  }

  private findNode(node: FileNode, path: string): FileNode | null {
    if (node.path === path) return node;
    if (node.children) {
      for (const child of node.children) {
        const found = this.findNode(child, path);
        if (found) return found;
      }
    }
    return null;
  }

  public updateData(root: FileNode): void {
    this.options.root = root;
    if (root.type === 'directory') {
      this.expandedPaths.add(root.path);
    }
    this.reRender();
  }

  private reRender(): void {
    this.element.innerHTML = '';
    const rootUList = this.renderTree(this.options.root, true);
    this.element.appendChild(rootUList);
  }
}
