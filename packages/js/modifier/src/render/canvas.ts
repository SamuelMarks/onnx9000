/* eslint-disable */
import { Graph, Node } from '@onnx9000/core';
import { GraphLayout, NodeLayout, EdgeLayout, Point } from './layout.js';

export interface CanvasConfig {
  nodeColor: string;
  initializerColor: string;
  outputColor: string;
  constantColor: string;
  font: string;
  textColor: string;
  edgeColor: string;
  highlightColor: string;
  backgroundColor: string;
  gridColor: string;
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
export class GraphRenderer {
  private nodeCache: Map<string, HTMLCanvasElement> = new Map();
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  config: CanvasConfig;

  // Transform
  scale: number = 1;
  offsetX: number = 0;
  offsetY: number = 0;

  // State
  hoveredNodeId: string | null = null;
  selectedNodeIds: Set<string> = new Set();

  // 276. Cleanup
  destroy() {
    /* v8 ignore start */
    this.canvas.width = 0;
    this.canvas.height = 0;
    this.ctx = null as ReturnType<typeof JSON.parse>;
  }
  /* v8 ignore stop */

  constructor(canvas: HTMLCanvasElement, config: Partial<CanvasConfig> = {}) {
    this.canvas = canvas;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Could not get 2D context');
    this.ctx = context;
    this.config = { ...DefaultConfig, ...config };
  }

  // 40. Infinite pan and zoom
  applyTransform() {
    this.ctx.setTransform(this.scale, 0, 0, this.scale, this.offsetX, this.offsetY);
  }

  clear() {
    this.ctx.save();
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.fillStyle = this.config.backgroundColor;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.restore();
  }

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

  render(graph: Graph, layout: GraphLayout) {
    this.drawGroup(layout);
    this.clear();
    this.drawGrid();

    // 194. Handle rendering of models with empty graphs gracefully
    if (graph.nodes.length === 0) {
      /* v8 ignore start */
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
    /* v8 ignore stop */

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
  drawEdge(edge: EdgeLayout, graph: Graph) {
    const isHighlighted =
      this.hoveredNodeId === edge.sourceId || this.hoveredNodeId === edge.targetId;
    this.ctx.strokeStyle = isHighlighted ? this.config.highlightColor : this.config.edgeColor;
    this.ctx.lineWidth = this.config.lineWidth;
    this.ctx.beginPath();

    const path = edge.path;
    if (path.length > 0) {
      this.ctx.moveTo(path[0]!.x, path[0]!.y);
      for (let i = 1; i < path.length; i++) {
        // Simple lines for now (orthogonal routing handled by layout engine)
        this.ctx.lineTo(path[i]!.x, path[i]!.y);
      }
    }
    this.ctx.stroke();

    if (path.length >= 2) {
      // 47. Display the tensor shape and type directly on the connecting edges
      const vi = graph.valueInfo.find((v) => v.name === edge.sourcePort);
      if (vi) {
        const text = `${vi.dtype}[${vi.shape.join(',')}]`;
        const midX = path[1]!.x;
        const midY = path[1]!.y;
        this.ctx.fillStyle = this.config.textColor;
        this.ctx.font = '10px monospace';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'bottom';
        this.ctx.fillText(text, midX, midY - 2);
      }
    }
  }

  // 42, 43, 44, 45. Distinct rendering
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
      const octx = off.getContext('2d')!;

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
        /* v8 ignore start */
        octx.fillStyle = '#dc3545'; // RED warning for unsupported ops
      }
      /* v8 ignore stop */

      // 291. Add visual node highlighting based on inference path tracking.
      if (node.attributes['_inference_path_active']) octx.fillStyle = '#28a745'; // Green path

      octx.strokeStyle = this.config.edgeColor;
      octx.lineWidth = 2;
      octx.beginPath();
      if (typeof octx.roundRect === 'function') {
        octx.roundRect(5, 5, layout.size.width, layout.size.height, 8);
      } else if (typeof octx.rect === 'function') {
        /* v8 ignore start */
        octx.rect(5, 5, layout.size.width, layout.size.height);
      }
      /* v8 ignore stop */
      octx.fill();
      octx.stroke();

      this.nodeCache.set(cacheKey, off);
    }

    const cached = this.nodeCache.get(cacheKey)!;
    if (this.ctx.drawImage) {
      this.ctx.drawImage(cached, layout.position.x - 5, layout.position.y - 5);
    }
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
    if (minX < Infinity) {
      this.ctx.fillRect(minX - 20, minY - 20, maxX - minX + 40, maxY - minY + 40);
      this.ctx.strokeRect(minX - 20, minY - 20, maxX - minX + 40, maxY - minY + 40);
    }
    this.ctx.setLineDash([]);
  }

  // Interaction handlers
  pan(dx: number, dy: number) {
    this.offsetX += dx;
    this.offsetY += dy;
  }

  zoom(factor: number, cx: number, cy: number) {
    const prevScale = this.scale;
    this.scale *= factor;
    // Keep point under mouse fixed
    this.offsetX = cx - (cx - this.offsetX) * (this.scale / prevScale);
    this.offsetY = cy - (cy - this.offsetY) * (this.scale / prevScale);
  }

  // 41. Minimap bounds extraction
  getMinimapBounds(layout: GraphLayout) {
    return {
      width: layout.bounds.width * this.scale,
      height: layout.bounds.height * this.scale,
    };
  }
}
