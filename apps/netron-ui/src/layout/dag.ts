/* eslint-disable no-control-regex */

/**
 * @fileoverview Defines data structures and algorithms for Directed Acyclic Graph (DAG) layout computation.
 * This module is responsible for computing the visual layout of an ONNX graph,
 * producing node and edge coordinates for either Top-Bottom (TB) or Left-Right (LR) directions.
 */
import type { Graph } from '@onnx9000/core';

/** Type alias `FlowDirection`. */
export type FlowDirection = 'TB' | 'LR';

/** Interface `Box` definition. */
export interface Box {
  /** Property `x`. */
  x: number;
  /** Property `y`. */
  y: number;
  /** Property `width`. */
  width: number;
  /** Property `height`. */
  height: number;
}

/** Interface `LayoutNode` definition. */
export interface LayoutNode extends Box {
  /** Property `id`. */
  id: string;
  /** Property `opType`. */
  opType: string;
  /** Property `name`. */
  name: string;
  /** Property `type`. */
  type: 'node' | 'input' | 'output' | 'constant';
  /** Property `stringValue`. */
  stringValue?: string; // 297. For inline string rendering
}

/** Interface `LayoutEdge` definition. */
export interface LayoutEdge {
  /** Property `from`. */
  from: string; // Node ID
  /** Property `to`. */
  to: string; // Node ID
  /** Property `points`. */
  points: { x: number; y: number }[];
  /** Property `tensorName`. */
  tensorName: string;
  /** Property `dtype`. */
  dtype?: string;
  /** Property `shape`. */
  shape?: string;
  /** Property `isOptional`. */
  isOptional?: boolean;
}

/** Interface `LayoutGroup` definition. */
export interface LayoutGroup extends Box {
  /** Property `name`. */
  name: string;
  /** Property `depth`. */
  depth: number;
}

/** Interface `GraphLayout` definition. */
export interface GraphLayout {
  /** Property `nodes`. */
  nodes: LayoutNode[];
  /** Property `edges`. */
  edges: LayoutEdge[];
  /** Property `groups`. */
  groups?: LayoutGroup[];
  /** Property `width`. */
  width: number;
  /** Property `height`. */
  height: number;
}

/**
 * Executes `computeLayout`.
 * @param graph Parameter `graph`.
 * @param direction Parameter `direction`.
 * @returns Result of `computeLayout`.
 */
export function computeLayout(graph: Graph, direction: FlowDirection = 'TB'): GraphLayout {
  const layoutNodes: LayoutNode[] = [];
  const layoutEdges: LayoutEdge[] = [];
  const layoutGroups: LayoutGroup[] = [];

  const producerMap = new Map<string, string>(); // tensorName -> producerNodeId
  const allNodeIds = new Set<string>();

  // Create synthetic nodes for inputs
  for (const input of graph.inputs) {
    if (graph.initializers.includes(input.name)) continue;
    producerMap.set(input.name, `input_${input.name}`);
    allNodeIds.add(`input_${input.name}`);
  }

  // Constants
  for (const init of graph.initializers) {
    producerMap.set(init, `const_${init}`);
    allNodeIds.add(`const_${init}`);
  }

  for (const node of graph.nodes) {
    allNodeIds.add(node.id);
    for (const out of node.outputs) {
      if (out) producerMap.set(out, node.id);
    }
  }

  // Create synthetic nodes for outputs
  for (const output of graph.outputs) {
    allNodeIds.add(`output_${output.name}`);
  }

  // 2. Assign topological levels
  const levels = new Map<string, number>();

  function getLevel(nodeId: string): number {
    if (levels.has(nodeId)) return levels.get(nodeId) as number;

    let maxParentLevel = -1;

    if (nodeId.startsWith('input_') || nodeId.startsWith('const_')) {
      maxParentLevel = -1;
    } else if (nodeId.startsWith('output_')) {
      const tensorName = nodeId.substring(7);
      const p = producerMap.get(tensorName);
      if (p) maxParentLevel = Math.max(maxParentLevel, getLevel(p));
    } else {
      const node = graph.nodes.find((n) => n.id === nodeId);
      if (node) {
        for (const input of node.inputs) {
          if (!input) continue;
          const p = producerMap.get(input);
          if (p) {
            maxParentLevel = Math.max(maxParentLevel, getLevel(p));
          }
        }
      }
    }

    const level = maxParentLevel + 1;
    levels.set(nodeId, level);
    return level;
  }

  let maxLevel = 0;
  for (const nodeId of allNodeIds) {
    maxLevel = Math.max(maxLevel, getLevel(nodeId));
  }

  // Group by levels
  const levelBuckets: string[][] = Array.from({ length: maxLevel + 1 }, () => []);
  for (const nodeId of allNodeIds) {
    levelBuckets[getLevel(nodeId)]?.push(nodeId);
  }

  const NODE_WIDTH = 120;
  const NODE_HEIGHT = 40;
  const HORIZONTAL_GAP = 50;
  const VERTICAL_GAP = 80;

  let totalWidth = 0;
  let totalHeight = 0;

  const positions = new Map<string, Box>();

  if (direction === 'TB') {
    let currentY = 50;
    for (const bucket of levelBuckets) {
      const bucketWidth = bucket.length * (NODE_WIDTH + HORIZONTAL_GAP) - HORIZONTAL_GAP;
      let currentX = -bucketWidth / 2;

      for (const nodeId of bucket) {
        let opType = 'Unknown';
        let name = nodeId;
        let type: LayoutNode['type'] = 'node';
        let stringValue: string | undefined;

        if (nodeId.startsWith('input_')) {
          opType = 'Input';
          name = nodeId.substring(6);
          type = 'input';
        } else if (nodeId.startsWith('const_')) {
          opType = 'Constant';
          name = nodeId.substring(6);
          type = 'constant';
          // 297. Find the string value if any
          const tensor = graph.tensors[name];
          if (tensor && tensor.dtype === 'string' && tensor.data) {
            const decoder = new TextDecoder('utf-8');
            const str = Array.from(decoder.decode(tensor.data))
              .filter((ch) => {
                const code = ch.charCodeAt(0);
                return code > 0x1f && code !== 0x7f;
              })
              .join('');
            stringValue = str.length > 15 ? `${str.substring(0, 15)}...` : str;
          }
        } else if (nodeId.startsWith('output_')) {
          opType = 'Output';
          name = nodeId.substring(7);
          type = 'output';
        } else {
          const n = graph.nodes.find((n) => n.id === nodeId);
          if (n) {
            opType = n.opType;
            name = n.name;
            if (opType === 'Constant' && n.attributes.value_string) {
              const v = String(n.attributes.value_string.value);
              stringValue = v.length > 15 ? `${v.substring(0, 15)}...` : v;
            }
          }
        }

        const dynamicWidth = Math.max(
          NODE_WIDTH,
          opType.length * 10 + (stringValue ? stringValue.length * 8 : 0) + 20,
        );
        const box: Box = {
          x: currentX,
          y: currentY,
          width: dynamicWidth,
          height: NODE_HEIGHT,
        };
        positions.set(nodeId, box);

        if (stringValue !== undefined) {
          layoutNodes.push({
            ...box,
            id: nodeId,
            opType,
            name,
            type,
            stringValue,
          });
        } else {
          layoutNodes.push({
            ...box,
            id: nodeId,
            opType,
            name,
            type,
          });
        }
        currentX += dynamicWidth + HORIZONTAL_GAP;
      }
      totalWidth = Math.max(totalWidth, bucketWidth);
      currentY += NODE_HEIGHT + VERTICAL_GAP;
    }
    totalHeight = currentY;
  } else {
    // LR
    let currentX = 50;
    for (const bucket of levelBuckets) {
      const bucketHeight = bucket.length * (NODE_HEIGHT + VERTICAL_GAP) - VERTICAL_GAP;
      let currentY = -bucketHeight / 2;

      for (const nodeId of bucket) {
        let opType = 'Unknown';
        let name = nodeId;
        let type: LayoutNode['type'] = 'node';
        let stringValue: string | undefined;

        if (nodeId.startsWith('input_')) {
          opType = 'Input';
          name = nodeId.substring(6);
          type = 'input';
        } else if (nodeId.startsWith('const_')) {
          opType = 'Constant';
          name = nodeId.substring(6);
          type = 'constant';
          const tensor = graph.tensors[name];
          if (tensor && tensor.dtype === 'string' && tensor.data) {
            const decoder = new TextDecoder('utf-8');
            const str = Array.from(decoder.decode(tensor.data))
              .filter((ch) => {
                const code = ch.charCodeAt(0);
                return code > 0x1f && code !== 0x7f;
              })
              .join('');
            stringValue = str.length > 15 ? `${str.substring(0, 15)}...` : str;
          }
        } else if (nodeId.startsWith('output_')) {
          opType = 'Output';
          name = nodeId.substring(7);
          type = 'output';
        } else {
          const n = graph.nodes.find((n) => n.id === nodeId);
          if (n) {
            opType = n.opType;
            name = n.name;
            if (opType === 'Constant' && n.attributes.value_string) {
              const v = String(n.attributes.value_string.value);
              stringValue = v.length > 15 ? `${v.substring(0, 15)}...` : v;
            }
          }
        }

        const dynamicWidth = Math.max(
          NODE_WIDTH,
          opType.length * 10 + (stringValue ? stringValue.length * 8 : 0) + 20,
        );
        const box: Box = {
          x: currentX,
          y: currentY,
          width: dynamicWidth,
          height: NODE_HEIGHT,
        };
        positions.set(nodeId, box);

        if (stringValue !== undefined) {
          layoutNodes.push({
            ...box,
            id: nodeId,
            opType,
            name,
            type,
            stringValue,
          });
        } else {
          layoutNodes.push({
            ...box,
            id: nodeId,
            opType,
            name,
            type,
          });
        }
        currentY += NODE_HEIGHT + VERTICAL_GAP;
      }
      totalHeight = Math.max(totalHeight, bucketHeight);
      currentX += NODE_WIDTH + HORIZONTAL_GAP;
    }
    totalWidth = currentX;
  }

  // Compute NameScope groups
  const scopeMap = new Map<
    string,
    {
      minX: number;
      minY: number;
      maxX: number;
      maxY: number;
      count: number;
      depth: number;
    }
  >();
  for (const node of layoutNodes) {
    if (node.type !== 'node') continue;
    const parts = node.name.split('/');
    if (parts.length > 1) {
      let currentScope = '';
      for (let i = 0; i < parts.length - 1; i++) {
        currentScope += (i === 0 ? '' : '/') + parts[i];
        if (!scopeMap.has(currentScope)) {
          scopeMap.set(currentScope, {
            minX: Infinity,
            minY: Infinity,
            maxX: -Infinity,
            maxY: -Infinity,
            count: 0,
            depth: i,
          });
        }
        const b = scopeMap.get(currentScope) as {
          minX: number;
          minY: number;
          maxX: number;
          maxY: number;
          count: number;
          depth: number;
        };
        b.minX = Math.min(b.minX, node.x);
        b.minY = Math.min(b.minY, node.y);
        b.maxX = Math.max(b.maxX, node.x + node.width);
        b.maxY = Math.max(b.maxY, node.y + node.height);
        b.count++;
      }
    }
  }

  for (const [scopeName, bounds] of scopeMap.entries()) {
    if (bounds.count > 1) {
      const padding = 20 + bounds.depth * 10;
      layoutGroups.push({
        name: scopeName,
        depth: bounds.depth,
        x: bounds.minX - padding,
        y: bounds.minY - padding - 20, // Extra top padding for label
        width: bounds.maxX - bounds.minX + padding * 2,
        height: bounds.maxY - bounds.minY + padding * 2 + 20,
      });
    }
  }

  // Sort groups by depth (deepest first, or shallowest first for rendering)
  layoutGroups.sort((a, b) => a.depth - b.depth);

  // Helper to add edge
  function addEdge(from: string, to: string, tensorName: string) {
    const fromBox = positions.get(from) as Box;
    const toBox = positions.get(to) as Box;

    // Attempt to format tensor shapes/dtypes
    let dtypeStr = '';
    let shapeStr = '';

    // Find info
    const info =
      graph.inputs.find((i) => i.name === tensorName) ||
      graph.outputs.find((o) => o.name === tensorName);
    if (info) {
      dtypeStr = info.dtype;
      shapeStr = `[${info.shape.join(', ')}]`;
    } else {
      const t = graph.tensors[tensorName];
      if (t) {
        dtypeStr = t.dtype;
        shapeStr = `[${t.shape.join(', ')}]`;
      }
    }

    if (direction === 'TB') {
      layoutEdges.push({
        from,
        to,
        tensorName,
        dtype: dtypeStr,
        shape: shapeStr,
        points: [
          {
            x: fromBox.x + fromBox.width / 2,
            y: fromBox.y + fromBox.height,
          },
          { x: toBox.x + toBox.width / 2, y: toBox.y },
        ],
      });
    } else {
      layoutEdges.push({
        from,
        to,
        tensorName,
        dtype: dtypeStr,
        shape: shapeStr,
        points: [
          {
            x: fromBox.x + fromBox.width,
            y: fromBox.y + fromBox.height / 2,
          },
          { x: toBox.x, y: toBox.y + toBox.height / 2 },
        ],
      });
    }
  }

  // Node edges
  for (const node of graph.nodes) {
    for (const input of node.inputs) {
      if (input === '') {
        // This is an omitted optional input, no edge to draw.
        continue;
      }
      const p = producerMap.get(input);
      if (p) addEdge(p, node.id, input);
    }
  }

  // Output edges
  for (const output of graph.outputs) {
    const p = producerMap.get(output.name);
    if (p) addEdge(p, `output_${output.name}`, output.name);
  }

  return {
    nodes: layoutNodes,
    edges: layoutEdges,
    groups: layoutGroups,
    width: totalWidth,
    height: totalHeight,
  };
}
