export class LayoutBuilder {
  container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  // 51. Build the main layout
  build() {
    this.container.style.display = 'flex';
    this.container.style.flexDirection = 'row';
    this.container.style.height = '100vh';
    this.container.style.width = '100vw';
    this.container.style.overflow = 'hidden';
    this.container.style.fontFamily = 'system-ui, -apple-system, sans-serif';
    this.container.style.backgroundColor = '#f8f9fa';

    // Left Panel (Structure Tree)
    const leftPanel = document.createElement('div');
    leftPanel.id = 'modifier-left-panel';
    leftPanel.setAttribute('data-testid', 'left-panel');
    leftPanel.style.width = '250px';
    leftPanel.style.minWidth = '200px';
    leftPanel.style.borderRight = '1px solid #dee2e6';
    leftPanel.style.backgroundColor = '#ffffff';
    leftPanel.style.display = 'flex';
    leftPanel.style.flexDirection = 'column';

    const treeHeader = document.createElement('div');
    treeHeader.textContent = 'Model Structure';
    treeHeader.style.padding = '12px';
    treeHeader.style.fontWeight = 'bold';
    treeHeader.style.borderBottom = '1px solid #dee2e6';

    const treeContent = document.createElement('div');
    treeContent.id = 'modifier-tree-content';
    treeContent.style.flex = '1';
    treeContent.style.overflowY = 'auto';
    treeContent.style.padding = '12px';

    leftPanel.appendChild(treeHeader);
    leftPanel.appendChild(treeContent);

    // Center Panel (Graph View)
    const centerPanel = document.createElement('div');
    centerPanel.id = 'modifier-center-panel';
    centerPanel.setAttribute('data-testid', 'center-panel');
    centerPanel.style.flex = '1';
    centerPanel.style.position = 'relative';
    centerPanel.style.overflow = 'hidden';

    // Right Panel (Properties)
    const rightPanel = document.createElement('div');
    rightPanel.id = 'modifier-right-panel';
    rightPanel.setAttribute('data-testid', 'right-panel');
    rightPanel.style.width = '300px';
    rightPanel.style.minWidth = '250px';
    rightPanel.style.borderLeft = '1px solid #dee2e6';
    rightPanel.style.backgroundColor = '#ffffff';
    rightPanel.style.display = 'flex';
    rightPanel.style.flexDirection = 'column';

    const propHeader = document.createElement('div');
    propHeader.textContent = 'Properties';
    propHeader.style.padding = '12px';
    propHeader.style.fontWeight = 'bold';
    propHeader.style.borderBottom = '1px solid #dee2e6';

    const propContent = document.createElement('div');
    propContent.id = 'modifier-properties-content';
    propContent.style.flex = '1';
    propContent.style.overflowY = 'auto';
    propContent.style.padding = '12px';

    rightPanel.appendChild(propHeader);
    rightPanel.appendChild(propContent);

    this.container.appendChild(leftPanel);
    this.container.appendChild(centerPanel);
    this.container.appendChild(rightPanel);

    return { leftPanel, centerPanel, rightPanel };
  }
}
