/**
 * @fileoverview canvas.ts
 * Provides canvas functionality for the modifier package.
 */

import type { Graph, Node } from '@onnx9000/core';
import { Onnx9000Error } from '@onnx9000/core';
import type { EdgeLayout, GraphLayout, NodeLayout } from './layout.js';

/** Interface `CanvasConfig` definition. */
export interface CanvasConfig {
  /** Property `nodeColor`. */
  nodeColor: string;
  /** Property `initializerColor`. */
  initializerColor: string;
  /** Property `outputColor`. */
  outputColor: string;
  /** Property `constantColor`. */
  constantColor: string;
  /** Property `font`. */
  font: string;
  /** Property `textColor`. */
  textColor: string;
  /** Property `edgeColor`. */
  edgeColor: string;
  /** Property `highlightColor`. */
  highlightColor: string;
  /** Property `backgroundColor`. */
  backgroundColor: string;
  /** Property `gridColor`. */
  gridColor: string;
  /** Property `lineWidth`. */
  lineWidth: number;
}

export const DefaultConfig: CanvasConfig = {
  nodeColor: '#f8f9fa',
  initializerColor: '#e9ecef',
  outputColor: '#ffe3e3',
  constantColor: '#e3f2fd',
  font: '12px monospace',
  textColor: '#212529',
  edgeColor: '#adb5bd',
  highlightColor: '#339af0',
  backgroundColor: '#ffffff',
  gridColor: '#f1f3f5',
  lineWidth: 2,
};

// 277. WebGL Instanced Drawing Hook / Fast Canvas caching
/** Represents the `GraphRenderer` class. */
export class GraphRenderer {
  /** Property `nodeCache` of `GraphRenderer`. */
  private nodeCache: Map<string, HTMLCanvasElement> = new Map();
  /** Property `canvas` of `GraphRenderer`. */
  canvas: HTMLCanvasElement;
  /** Property `ctx` of `GraphRenderer`. */
  ctx: CanvasRenderingContext2D;
  /** Property `config` of `GraphRenderer`. */
  config: CanvasConfig;

  // Transform
  /** Property `scale` of `GraphRenderer`. */
  scale: number = 1;
  /** Property `offsetX` of `GraphRenderer`. */
  offsetX: number = 0;
  /** Property `offsetY` of `GraphRenderer`. */
  offsetY: number = 0;

  // State
  /** Property `hoveredNodeId` of `GraphRenderer`. */
  hoveredNodeId: string | null = null;
  /** Property `selectedNodeIds` of `GraphRenderer`. */
  selectedNodeIds: Set<string> = new Set();

  // 276. Cleanup
  /**
   * Executes `destroy` on `GraphRenderer`.
   * @returns Result of `destroy`.
   */
  destroy() {
    this.canvas.width = 0;
    this.canvas.height = 0;
    this.ctx = null as ReturnType<typeof JSON.parse>;
  }

  /**
   * Initializes a new `GraphRenderer` instance.
   * @param canvas Parameter `canvas`.
   * @param config Parameter `config`.
   */
  constructor(canvas: HTMLCanvasElement, config: Partial<CanvasConfig> = {}) {
    this.canvas = canvas;
    const context = canvas.getContext('2d');
    if (!context) throw new Onnx9000Error('Could not get 2D context');
    this.ctx = context;
    this.config = { ...DefaultConfig, ...config };
  }

  // 40. Infinite pan and zoom
  /**
   * Executes `applyTransform` on `GraphRenderer`.
   * @returns Result of `applyTransform`.
   */
  applyTransform() {
    this.ctx.setTransform(this.scale, 0, 0, this.scale, this.offsetX, this.offsetY);
  }

  /**
   * Executes `clear` on `GraphRenderer`.
   * @returns Result of `clear`.
   */
  clear() {
    this.ctx.save();
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.fillStyle = this.config.backgroundColor;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.restore();
  }

  /**
   * Executes `drawGrid` on `GraphRenderer`.
   * @returns Result of `drawGrid`.
   */
  drawGrid() {
    this.ctx.save();
    this.applyTransform();
    this.ctx.strokeStyle = this.config.gridColor;
    this.ctx.lineWidth = 1 / this.scale;

    const step = 50;
    const startX = -this.offsetX / this.scale;
    const startY = -this.offsetY / this.scale;
    const endX = startX + this.canvas.width / this.scale;
    const endY = startY + this.canvas.height / this.scale;

    this.ctx.beginPath();
    for (let x = Math.floor(startX / step) * step; x < endX; x += step) {
      this.ctx.moveTo(x, startY);
      this.ctx.lineTo(x, endY);
    }
    for (let y = Math.floor(startY / step) * step; y < endY; y += step) {
      this.ctx.moveTo(startX, y);
      this.ctx.lineTo(endX, y);
    }
    this.ctx.stroke();
    this.ctx.restore();
  }

  /**
   * Executes `render` on `GraphRenderer`.
   * @param graph Parameter `graph`.
   * @param layout Parameter `layout`.
   * @returns Result of `render`.
   */
  render(graph: Graph, layout: GraphLayout) {
    this.drawGroup(layout);
    this.clear();
    this.drawGrid();

    // 194. Handle rendering of models with empty graphs gracefully
    if (graph.nodes.length === 0) {
      this.ctx.save();
      this.applyTransform();
      this.ctx.fillStyle = '#6c757d';
      this.ctx.font = '20px sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(
        'Empty Graph',
        this.canvas.width / 2 / this.scale,
        this.canvas.height / 2 / this.scale,
      );
      this.ctx.restore();
      return;
    }

    this.ctx.save();
    this.applyTransform();

    // Draw Edges
    for (const edge of layout.edges) {
      this.drawEdge(edge, graph);
    }

    // Draw Nodes
    for (const [id, nodeLayout] of Array.from(layout.nodes.entries())) {
      const nodeDef = graph.nodes.find((n) => n.id === id);
      if (nodeDef) {
        this.drawNode(nodeDef, nodeLayout, graph);
      }
    }

    this.ctx.restore();
  }

  // 48. Edge routing
  /**
   * Executes `drawEdge` on `GraphRenderer`.
   * @param edge Parameter `edge`.
   * @param graph Parameter `graph`.
   * @returns Result of `drawEdge`.
   */
  drawEdge(edge: EdgeLayout, graph: Graph) {
    const isHighlighted =
      this.hoveredNodeId === edge.sourceId || this.hoveredNodeId === edge.targetId;
    this.ctx.strokeStyle = isHighlighted ? this.config.highlightColor : this.config.edgeColor;
    this.ctx.lineWidth = this.config.lineWidth;
    this.ctx.beginPath();

    const path = edge.path;
    if (path.length > 0) {
      const p0 = path[0] as (typeof path)[number];
      this.ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < path.length; i++) {
        const pi = path[i] as (typeof path)[number];
        this.ctx.lineTo(pi.x, pi.y);
      }
    }
    this.ctx.stroke();

    if (path.length >= 2) {
      // 47. Display the tensor shape and type directly on the connecting edges
      const vi = graph.valueInfo.find((v) => v.name === edge.sourcePort);
      if (vi) {
        const p1 = path[1] as (typeof path)[number];
        const text = `${vi.dtype}[${vi.shape.join(',')}]`;
        this.ctx.fillStyle = this.config.textColor;
        this.ctx.font = '10px monospace';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'bottom';
        this.ctx.fillText(text, p1.x, p1.y - 2);
      }
    }
  }

  // 42, 43, 44, 45. Distinct rendering
  /**
   * Executes `drawNode` on `GraphRenderer`.
   * @param node Parameter `node`.
   * @param layout Parameter `layout`.
   * @param graph Parameter `graph`.
   * @returns Result of `drawNode`.
   */
  drawNode(node: Node, layout: NodeLayout, graph: Graph) {
    // 277. Pseudo-instancing via offscreen canvas caching
    const cacheKey =
      node.opType +
      (this.hoveredNodeId === node.id ? '_hover' : '') +
      (this.selectedNodeIds.has(node.id) ? '_sel' : '');

    // We only cache the background shape/colors, text is dynamic
    if (!this.nodeCache.has(cacheKey)) {
      const off = document.createElement('canvas');
      off.width = layout.size.width + 10;
      off.height = layout.size.height + 10;
      const octx = off.getContext('2d') as CanvasRenderingContext2D;

      octx.fillStyle = this.selectedNodeIds.has(node.id)
        ? this.config.highlightColor
        : this.config.nodeColor;
      if (this.hoveredNodeId === node.id) octx.fillStyle = '#ffc107'; // hover color

      // 191. Create custom warning badges on nodes that are known to be slow or unsupported in WebGPU
      if (
        node.opType === 'NonMaxSuppression' ||
        node.opType === 'TopK' ||
        node.opType === 'RoiAlign'
      ) {
        octx.fillStyle = '#dc3545'; // RED warning for unsupported ops
      }

      // 291. Add visual node highlighting based on inference path tracking.
      if (node.attributes._inference_path_active) octx.fillStyle = '#28a745'; // Green path

      octx.strokeStyle = this.config.edgeColor;
      octx.lineWidth = 2;
      octx.beginPath();
      if (typeof octx.roundRect === 'function') {
        octx.roundRect(5, 5, layout.size.width, layout.size.height, 8);
      } else {
        octx.rect(5, 5, layout.size.width, layout.size.height);
      }

      octx.fill();
      octx.stroke();

      this.nodeCache.set(cacheKey, off);
    }

    const cached = this.nodeCache.get(cacheKey) as HTMLCanvasElement;
    this.ctx.drawImage(cached, layout.position.x - 5, layout.position.y - 5);
    const isSelected = this.selectedNodeIds.has(node.id);
    const isHovered = this.hoveredNodeId === node.id;

    // Determine type
    let fillStyle = this.config.nodeColor;
    if (node.opType === 'Constant') {
      fillStyle = this.config.constantColor;
    } else if (graph.outputs.some((o) => node.outputs.includes(o.name))) {
      fillStyle = this.config.outputColor;
    }

    if (isSelected || isHovered) {
      this.ctx.strokeStyle = this.config.highlightColor;
      this.ctx.lineWidth = 3;
    } else {
      this.ctx.strokeStyle = this.config.edgeColor;
      this.ctx.lineWidth = this.config.lineWidth;
    }

    this.ctx.fillStyle = fillStyle;

    // Draw rounded rect
    const { x, y } = layout.position;
    const { width, height } = layout.size;
    const radius = 8;

    this.ctx.beginPath();
    this.ctx.moveTo(x + radius, y);
    this.ctx.lineTo(x + width - radius, y);
    this.ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    this.ctx.lineTo(x + width, y + height - radius);
    this.ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    this.ctx.lineTo(x + radius, y + height);
    this.ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    this.ctx.lineTo(x, y + radius);
    this.ctx.quadraticCurveTo(x, y, x + radius, y);
    this.ctx.closePath();
    this.ctx.fill();
    this.ctx.stroke();

    // 46. Display op_type prominently
    this.ctx.fillStyle = this.config.textColor;
    this.ctx.font = this.config.font;
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(node.opType, x + width / 2, y + height / 2);
  }

  // 50. Implement visual grouping
  /**
   * Executes `drawGroup` on `GraphRenderer`.
   * @param layout Parameter `layout`.
   * @returns Result of `drawGroup`.
   */
  drawGroup(layout: GraphLayout) {
    if (layout.nodes.size === 0) return;
    this.ctx.fillStyle = 'rgba(200, 200, 200, 0.1)';
    this.ctx.strokeStyle = this.config.edgeColor;
    this.ctx.setLineDash([5, 5]);
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (const n of Array.from(layout.nodes.values())) {
      if (n.position.x < minX) minX = n.position.x;
      if (n.position.y < minY) minY = n.position.y;
      if (n.position.x + n.size.width > maxX) maxX = n.position.x + n.size.width;
      if (n.position.y + n.size.height > maxY) maxY = n.position.y + n.size.height;
    }
    this.ctx.fillRect(minX - 20, minY - 20, maxX - minX + 40, maxY - minY + 40);
    this.ctx.strokeRect(minX - 20, minY - 20, maxX - minX + 40, maxY - minY + 40);
    this.ctx.setLineDash([]);
  }

  // Interaction handlers
  /**
   * Executes `pan` on `GraphRenderer`.
   * @param dx Parameter `dx`.
   * @param dy Parameter `dy`.
   * @returns Result of `pan`.
   */
  pan(dx: number, dy: number) {
    this.offsetX += dx;
    this.offsetY += dy;
  }

  /**
   * Executes `zoom` on `GraphRenderer`.
   * @param factor Parameter `factor`.
   * @param cx Parameter `cx`.
   * @param cy Parameter `cy`.
   * @returns Result of `zoom`.
   */
  zoom(factor: number, cx: number, cy: number) {
    const prevScale = this.scale;
    this.scale *= factor;
    // Keep point under mouse fixed
    this.offsetX = cx - (cx - this.offsetX) * (this.scale / prevScale);
    this.offsetY = cy - (cy - this.offsetY) * (this.scale / prevScale);
  }

  // 41. Minimap bounds extraction
  /**
   * Executes `getMinimapBounds` on `GraphRenderer`.
   * @param layout Parameter `layout`.
   * @returns Result of `getMinimapBounds`.
   */
  getMinimapBounds(layout: GraphLayout) {
    return {
      width: layout.bounds.width * this.scale,
      height: layout.bounds.height * this.scale,
    };
  }
}
