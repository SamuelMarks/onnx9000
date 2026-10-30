/**
 * @fileoverview layout.ts
 * Provides layout functionality for the modifier package.
 */
import type { Graph, Node } from '@onnx9000/core';

/** Interface `Point` definition. */
export interface Point {
  /** Property `x`. */
  x: number;
  /** Property `y`. */
  y: number;
}

/** Interface `Size` definition. */
export interface Size {
  /** Property `width`. */
  width: number;
  /** Property `height`. */
  height: number;
}

/** Interface `NodeLayout` definition. */
export interface NodeLayout {
  /** Property `id`. */
  id: string;
  /** Property `position`. */
  position: Point;
  /** Property `size`. */
  size: Size;
  /** Property `layer`. */
  layer: number;
}

/** Interface `EdgeLayout` definition. */
export interface EdgeLayout {
  /** Property `sourceId`. */
  sourceId: string;
  /** Property `targetId`. */
  targetId: string;
  /** Property `sourcePort`. */
  sourcePort: string;
  /** Property `targetPort`. */
  targetPort: string;
  /** Property `path`. */
  path: Point[];
}

/** Interface `GraphLayout` definition. */
export interface GraphLayout {
  /** Property `nodes`. */
  nodes: Map<string, NodeLayout>;
  /** Property `edges`. */
  edges: EdgeLayout[];
  /** Property `bounds`. */
  bounds: Size;
}

/** Type alias `LayoutDirection`. */
export type LayoutDirection = 'TB' | 'LR';

/** Represents the `DagreLayoutEngine` class. */
export class DagreLayoutEngine {
  /** Property `direction` of `DagreLayoutEngine`. */
  direction: LayoutDirection;

  /**
   * Initializes a new `DagreLayoutEngine` instance.
   * @param direction Parameter `direction`.
   */
  constructor(direction: LayoutDirection = 'TB') {
    this.direction = direction;
  }

  // 271. Fallback to a fast grid layout
  /**
   * Executes `computeGrid` on `DagreLayoutEngine`.
   * @param graph Parameter `graph`.
   * @returns Result of `computeGrid`.
   */
  computeGrid(graph: Graph): GraphLayout {
    const nodes = new Map<string, NodeLayout>();
    const edges: EdgeLayout[] = [];

    const nodeWidth = 120;
    const nodeHeight = 40;
    const xSpacing = 40;
    const ySpacing = 60;

    let maxX = 0;
    let maxY = 0;

    const cols = Math.ceil(Math.sqrt(graph.nodes.length));

    for (let i = 0; i < graph.nodes.length; i++) {
      const node = graph.nodes[i] as Node;
      const r = Math.floor(i / cols);
      const c = i % cols;
      const x = c * (nodeWidth + xSpacing);
      const y = r * (nodeHeight + ySpacing);

      nodes.set(node.id, {
        id: node.id,
        position: { x, y },
        size: { width: nodeWidth, height: nodeHeight },
        layer: r,
      });

      if (x + nodeWidth > maxX) maxX = x + nodeWidth;
      if (y + nodeHeight > maxY) maxY = y + nodeHeight;
    }

    return {
      nodes,
      edges, // Edge routing is skipped/trivial in fallback grid to save time
      bounds: { width: maxX, height: maxY },
    };
  }

  // 37. Implement the Sugiyama layout algorithm (or integrate Dagre.js)
  // For a 0 dependency approach, we implement a lightweight topological layer assignment
  /**
   * Executes `compute` on `DagreLayoutEngine`.
   * @param graph Parameter `graph`.
   * @returns Result of `compute`.
   */
  compute(graph: Graph): GraphLayout {
    // 271. Fallback to a fast grid layout if Sugiyama fails or times out
    const startTime = performance.now();
    const nodes = new Map<string, NodeLayout>();
    const edges: EdgeLayout[] = [];

    // 1. Assign Layers (Longest Path)
    const layers = new Map<string, number>();
    const producerMap = new Map<string, Node>();

    for (const node of graph.nodes) {
      for (const out of node.outputs) {
        producerMap.set(out, node);
      }
    }

    const getLayer = (nodeId: string): number => {
      if (layers.has(nodeId)) return layers.get(nodeId) as number;
      const node = graph.nodes.find((n) => n.id === nodeId);
      if (!node) return 0;

      let maxLayer = -1;
      for (const inp of node.inputs) {
        const prod = producerMap.get(inp);
        if (prod) {
          const l = getLayer(prod.id);
          if (l > maxLayer) maxLayer = l;
        }
      }
      const myLayer = maxLayer + 1;
      layers.set(nodeId, myLayer);
      return myLayer;
    };

    let maxGraphLayer = 0;
    for (const node of graph.nodes) {
      // Timeout check
      if (performance.now() - startTime > 2000) {
        console.warn('Layout timeout Falling back to grid layout.');
        return this.computeGrid(graph);
      }

      const l = getLayer(node.id);
      if (l > maxGraphLayer) maxGraphLayer = l;
    }

    // 2. Layout nodes per layer
    const layerMap = new Map<number, Node[]>();
    for (const node of graph.nodes) {
      const l = layers.get(node.id) as number;
      if (!layerMap.has(l)) layerMap.set(l, []);
      layerMap.get(l)?.push(node);
    }

    const nodeWidth = 120;
    const nodeHeight = 40;
    const xSpacing = 40;
    const ySpacing = 60;

    let maxX = 0;
    let maxY = 0;

    for (let l = 0; l <= maxGraphLayer; l++) {
      const layerNodes = layerMap.get(l) || [];
      let currentX = 0;
      let currentY = 0;

      for (let i = 0; i < layerNodes.length; i++) {
        const node = layerNodes[i] as Node;

        // 38. Vertical mode (TB)
        // 39. Horizontal mode (LR)
        if (this.direction === 'TB') {
          currentX = i * (nodeWidth + xSpacing);
          currentY = l * (nodeHeight + ySpacing);
        } else {
          currentX = l * (nodeWidth + xSpacing);
          currentY = i * (nodeHeight + ySpacing);
        }

        // 265. Allow pinning specific nodes to fixed coordinates on the canvas
        if (node.attributes.pinned) {
          const px = node.attributes.pinned_x;
          const py = node.attributes.pinned_y;
          if (px && py) {
            currentX = px.value as number;
            currentY = py.value as number;
          }
        }

        // 292. Add custom layout padding configurations
        currentX += 20; // global padding
        currentY += 20;

        nodes.set(node.id, {
          id: node.id,
          position: { x: currentX, y: currentY },
          size: { width: nodeWidth, height: nodeHeight },
          layer: l,
        });

        if (currentX + nodeWidth > maxX) maxX = currentX + nodeWidth;
        if (currentY + nodeHeight > maxY) maxY = currentY + nodeHeight;
      }
    }

    // 3. Simple orthogonal edge routing
    for (const node of graph.nodes) {
      for (const inp of node.inputs) {
        const prod = producerMap.get(inp);
        if (prod) {
          const sourceLayout = nodes.get(prod.id) as NodeLayout;
          const targetLayout = nodes.get(node.id) as NodeLayout;

          let path: Point[] = [];

          if (this.direction === 'TB') {
            const startX = sourceLayout.position.x + sourceLayout.size.width / 2;
            const startY = sourceLayout.position.y + sourceLayout.size.height;
            const endX = targetLayout.position.x + targetLayout.size.width / 2;
            const endY = targetLayout.position.y;

            path = [
              { x: startX, y: startY },
              { x: startX, y: startY + (endY - startY) / 2 },
              { x: endX, y: startY + (endY - startY) / 2 },
              { x: endX, y: endY },
            ];
          } else {
            const startX = sourceLayout.position.x + sourceLayout.size.width;
            const startY = sourceLayout.position.y + sourceLayout.size.height / 2;
            const endX = targetLayout.position.x;
            const endY = targetLayout.position.y + targetLayout.size.height / 2;

            path = [
              { x: startX, y: startY },
              { x: startX + (endX - startX) / 2, y: startY },
              { x: startX + (endX - startX) / 2, y: endY },
              { x: endX, y: endY },
            ];
          }

          edges.push({
            sourceId: prod.id,
            targetId: node.id,
            sourcePort: inp,
            targetPort: inp,
            path,
          });
        }
      }
    }

    return {
      nodes,
      edges,
      bounds: { width: maxX, height: maxY },
    };
  }
}
