/**
 * @fileoverview FileTree.ts
 * Provides FileTree functionality for the Sphinx Demo UI.
 */
import { Component } from '../core/Component';

/** Interface `FileNode` definition. */
export interface FileNode {
  /** Property `name`. */
  name: string;
  /** Property `type`. */
  type: 'file' | 'directory';
  /** Property `path`. */
  path: string; // The full path
  /** Property `content`. */
  content?: string; // Optional raw file content for mock data
  /** Property `children`. */
  children?: FileNode[];
}

/** Interface `FileTreeOptions` definition. */
export interface FileTreeOptions {
  /** Property `root`. */
  root: FileNode;
  /** Property `onSelect`. */
  onSelect?: (path: string) => void;
}

/**
 * A highly reusable, nested file tree component for the Demo UI.
 */
export class FileTree extends Component<HTMLDivElement> {
  /** Property `options` of `FileTree`. */
  public options: FileTreeOptions;
  /** Property `selectedPath` of `FileTree`. */
  private selectedPath: string | null = null;
  /** Property `expandedPaths` of `FileTree`. */
  private expandedPaths: Set<string> = new Set();

  /**
   * Initializes a new `FileTree` instance.
   * @param options Parameter `options`.
   */
  constructor(options: FileTreeOptions) {
    super();
    this.options = options;

    // Auto-expand root by default
    if (this.options.root.type === 'directory') {
      this.expandedPaths.add(this.options.root.path);
    }

    this.element = this.render();
  }

  /**
   * Executes `render` on `FileTree`.
   * @returns Result of `render`.
   */
  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-file-tree';
    container.setAttribute('role', 'tree');

    const rootUList = this.renderTree(this.options.root, true);
    container.appendChild(rootUList);

    return container;
  }

  /**
   * Executes `renderTree` on `FileTree`.
   * @param node Parameter `node`.
   * @param isRoot Parameter `isRoot`.
   * @returns Result of `renderTree`.
   */
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

  /** Executes `onMount` on `FileTree`. */
  protected onMount(): void {
    this.addDOMListener(this.element, 'click', (e) => {
      const target = e.target as HTMLElement;
      const label = target.closest('.demo-file-tree-label');
      if (!label) return;

      const li = label.closest('.demo-file-tree-node') as HTMLElement;
      const path = li.getAttribute('data-path') as string;
      const node = this.findNode(this.options.root, path) as FileNode;

      if (node.type === 'directory') {
        this.toggleDirectory(node.path);
      } else {
        this.selectFile(node.path);
      }
    });
  }

  /**
   * Executes `toggleDirectory` on `FileTree`.
   * @param path Parameter `path`.
   */
  private toggleDirectory(path: string): void {
    if (this.expandedPaths.has(path)) {
      this.expandedPaths.delete(path);
    } else {
      this.expandedPaths.add(path);
    }
    this.reRender();
  }

  /**
   * Executes `getSelectedPath` on `FileTree`.
   * @returns Result of `getSelectedPath`.
   */
  public getSelectedPath(): string | null {
    return this.selectedPath;
  }

  /**
   * Executes `selectFile` on `FileTree`.
   * @param path Parameter `path`.
   */
  public selectFile(path: string): void {
    this.selectedPath = path;
    this.reRender();

    if (this.options.onSelect) {
      this.options.onSelect(path);
    }
  }

  /**
   * Executes `findNode` on `FileTree`.
   * @param node Parameter `node`.
   * @param path Parameter `path`.
   * @returns Result of `findNode`.
   */
  public findNode(node: FileNode, path: string): FileNode | null {
    if (node.path === path) return node;
    if (node.children) {
      for (const child of node.children) {
        const found = this.findNode(child, path);
        if (found) return found;
      }
    }
    return null;
  }

  /**
   * Executes `updateData` on `FileTree`.
   * @param root Parameter `root`.
   */
  public updateData(root: FileNode): void {
    this.options.root = root;
    if (root.type === 'directory') {
      this.expandedPaths.add(root.path);
    }
    this.reRender();
  }

  /** Executes `reRender` on `FileTree`. */
  private reRender(): void {
    this.element.innerHTML = '';
    const rootUList = this.renderTree(this.options.root, true);
    this.element.appendChild(rootUList);
  }
}
