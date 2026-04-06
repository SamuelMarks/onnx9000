/* eslint-disable */
// @ts-nocheck
import { Component } from '../core/Component';
import cytoscape from 'cytoscape';
import { OnnxAdapter, VizGraph } from '../core/OnnxAdapter';
import { globalEventBus } from '../core/EventBus';

export class OnnxVisualizer extends Component<HTMLDivElement> {
  private cy: cytoscape.Core | null = null;
  private tooltip!: HTMLDivElement;

  constructor() {
    super();
    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-onnx-viz-container';
    container.style.width = '100%';
    container.style.height = '100%';
    container.style.position = 'relative';
    container.style.backgroundColor = 'var(--bg-color-panel)';

    this.tooltip = document.createElement('div');
    this.tooltip.className = 'demo-onnx-tooltip';
    this.tooltip.style.position = 'absolute';
    this.tooltip.style.display = 'none';
    this.tooltip.style.backgroundColor = 'var(--bg-color-main)';
    this.tooltip.style.border = '1px solid var(--border-color)';
    this.tooltip.style.padding = '8px';
    this.tooltip.style.borderRadius = '4px';
    this.tooltip.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
    this.tooltip.style.zIndex = '10';
    this.tooltip.style.pointerEvents = 'none';
    this.tooltip.style.fontFamily = 'var(--font-mono)';
    this.tooltip.style.fontSize = '0.8rem';

    container.appendChild(this.tooltip);

    return container;
  }

  protected onMount(): void {
    (window as object).__CY__ = this.cy = cytoscape({
      container: this.element,
      elements: [],
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            'background-color': '#1c7ed6',
            color: '#fff',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '10px',
            shape: 'round-rectangle',
            width: 'label',
            height: 'label',
            padding: '10px'
          }
        },
        {
          selector: '.onnx-input, .onnx-output',
          style: {
            'background-color': '#2b8a3e',
            shape: 'ellipse'
          }
        },
        {
          selector: '.onnx-initializer',
          style: {
            'background-color': '#e67700',
            shape: 'barrel'
          }
        },
        {
          selector: 'edge',
          style: {
            width: 2,
            'line-color': '#adb5bd',
            'target-arrow-color': '#adb5bd',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            label: 'data(label)',
            'font-size': '8px',
            color: '#868e96',
            'text-background-color': '#ffffff',
            'text-background-opacity': 1,
            'text-background-padding': '2px'
          }
        }
      ],
      layout: {
        name: 'preset'
      }
    });

    this.cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      const data = node.data();

      this.tooltip.innerHTML = `
        <strong>${data.type.toUpperCase()}: ${data.label}</strong>
        ${data.dtype ? `<br/>Type: ${data.dtype}` : ''}
        ${data.attributes ? `<br/>Attrs: ${JSON.stringify(data.attributes)}` : ''}
      `;

      const pos = evt.renderedPosition || { x: 0, y: 0 };
      this.tooltip.style.left = `${pos.x + 15}px`;
      this.tooltip.style.top = `${pos.y + 15}px`;
      this.tooltip.style.display = 'block';
    });

    this.cy.on('tap', (evt) => {
      if (evt.target === this.cy) {
        this.tooltip.style.display = 'none';
      }
    });

    this.onCleanup(() => {
      if (this.cy) {
        this.cy.destroy();
      }
    });

    // Listen to global graph update events from WASM
    this.onCleanup(
      globalEventBus.on<VizGraph>('ONNX_GRAPH_GENERATED', (graph) => {
        this.renderGraph(graph);
      })
    );

    this.onCleanup(
      globalEventBus.on<string>('TAB_CHANGED', (tabId) => {
        if (tabId === 'viz' && this.cy) {
          setTimeout(() => {
            this.cy!.resize();
            this.cy!.fit();
            this.cy!.fit();
          }, 50);
        }
      })
    );
  }

  public renderGraph(graph: VizGraph | null): void {
    if (!this.cy) return;

    if (!graph) {
      this.cy.elements().remove();
      this.tooltip.style.display = 'none';
      return;
    }

    const elements = OnnxAdapter.toCytoscape(graph);

    this.cy.elements().remove();
    this.cy.add(elements as object);

    // Simple top-to-bottom layout for DAGs
    this.cy
      .layout({
        name: 'breadthfirst',
        directed: true,
        spacingFactor: 1.5,
        fit: true,
        padding: 50
      })
      .run();

    this.cy.center();
  }
}
