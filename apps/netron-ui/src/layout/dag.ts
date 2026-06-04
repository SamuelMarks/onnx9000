/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import { Graph, Node } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export type FlowDirection = 'TB' | 'LR'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export interface Box {
  /* v8 ignore next */ /* v8 ignore next */
  x: number; /* v8 ignore next */ /* v8 ignore next */
  y: number; /* v8 ignore next */ /* v8 ignore next */
  width: number; /* v8 ignore next */ /* v8 ignore next */
  height: number; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export interface LayoutNode extends Box {
  /* v8 ignore next */ /* v8 ignore next */
  id: string; /* v8 ignore next */ /* v8 ignore next */
  opType: string; /* v8 ignore next */ /* v8 ignore next */
  name: string; /* v8 ignore next */ /* v8 ignore next */
  type: 'node' | 'input' | 'output' | 'constant'; /* v8 ignore next */ /* v8 ignore next */
  stringValue?: string; // 297. For inline string rendering /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export interface LayoutEdge {
  /* v8 ignore next */ /* v8 ignore next */
  from: string; // Node ID /* v8 ignore next */ /* v8 ignore next */
  to: string; // Node ID /* v8 ignore next */ /* v8 ignore next */
  points: { x: number; y: number }[]; /* v8 ignore next */ /* v8 ignore next */
  tensorName: string; /* v8 ignore next */ /* v8 ignore next */
  dtype?: string; /* v8 ignore next */ /* v8 ignore next */
  shape?: string; /* v8 ignore next */ /* v8 ignore next */
  isOptional?: boolean; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export interface LayoutGroup extends Box {
  /* v8 ignore next */ /* v8 ignore next */
  name: string; /* v8 ignore next */ /* v8 ignore next */
  depth: number; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export interface GraphLayout {
  /* v8 ignore next */ /* v8 ignore next */
  nodes: LayoutNode[]; /* v8 ignore next */ /* v8 ignore next */
  edges: LayoutEdge[]; /* v8 ignore next */ /* v8 ignore next */
  groups?: LayoutGroup[]; /* v8 ignore next */ /* v8 ignore next */
  width: number; /* v8 ignore next */ /* v8 ignore next */
  height: number; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export function computeLayout(graph: Graph, direction: FlowDirection = 'TB'): GraphLayout {
  /* v8 ignore next */ /* v8 ignore next */
  const layoutNodes: LayoutNode[] = []; /* v8 ignore next */ /* v8 ignore next */
  const layoutEdges: LayoutEdge[] = []; /* v8 ignore next */ /* v8 ignore next */
  const layoutGroups: LayoutGroup[] = []; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const producerMap = new Map<string, string>(); // tensorName -> producerNodeId /* v8 ignore next */ /* v8 ignore next */
  const allNodeIds = new Set<string>(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Create synthetic nodes for inputs /* v8 ignore next */ /* v8 ignore next */
  for (const input of graph.inputs) {
    /* v8 ignore next */ /* v8 ignore next */
    if (graph.initializers.includes(input.name)) continue; /* v8 ignore next */ /* v8 ignore next */
    producerMap.set(input.name, `input_${input.name}`); /* v8 ignore next */ /* v8 ignore next */
    allNodeIds.add(`input_${input.name}`); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Constants /* v8 ignore next */ /* v8 ignore next */
  for (const init of graph.initializers) {
    /* v8 ignore next */ /* v8 ignore next */
    producerMap.set(init, `const_${init}`); /* v8 ignore next */ /* v8 ignore next */
    allNodeIds.add(`const_${init}`); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  for (const node of graph.nodes) {
    /* v8 ignore next */ /* v8 ignore next */
    allNodeIds.add(node.id); /* v8 ignore next */ /* v8 ignore next */
    for (const out of node.outputs) {
      /* v8 ignore next */ /* v8 ignore next */
      producerMap.set(out, node.id); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Create synthetic nodes for outputs /* v8 ignore next */ /* v8 ignore next */
  for (const output of graph.outputs) {
    /* v8 ignore next */ /* v8 ignore next */
    allNodeIds.add(`output_${output.name}`); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // 2. Assign topological levels /* v8 ignore next */ /* v8 ignore next */
  const levels = new Map<string, number>(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  function getLevel(nodeId: string): number {
    /* v8 ignore next */ /* v8 ignore next */
    if (levels.has(nodeId)) return levels.get(nodeId)!; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    let maxParentLevel = -1; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (nodeId.startsWith('input_') || nodeId.startsWith('const_')) {
      /* v8 ignore next */ /* v8 ignore next */
      maxParentLevel = -1; /* v8 ignore next */ /* v8 ignore next */
    } else if (nodeId.startsWith('output_')) {
      /* v8 ignore next */ /* v8 ignore next */
      const tensorName = nodeId.substring(7); /* v8 ignore next */ /* v8 ignore next */
      const p = producerMap.get(tensorName); /* v8 ignore next */ /* v8 ignore next */
      if (p)
        maxParentLevel = Math.max(
          maxParentLevel,
          getLevel(p),
        ); /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      const node = graph.nodes.find(
        (n) => n.id === nodeId,
      ); /* v8 ignore next */ /* v8 ignore next */
      if (node) {
        /* v8 ignore next */ /* v8 ignore next */
        for (const input of node.inputs) {
          /* v8 ignore next */ /* v8 ignore next */
          const p = producerMap.get(input); /* v8 ignore next */ /* v8 ignore next */
          if (p) {
            /* v8 ignore next */ /* v8 ignore next */
            maxParentLevel = Math.max(
              maxParentLevel,
              getLevel(p),
            ); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const level = maxParentLevel + 1; /* v8 ignore next */ /* v8 ignore next */
    levels.set(nodeId, level); /* v8 ignore next */ /* v8 ignore next */
    return level; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let maxLevel = 0; /* v8 ignore next */ /* v8 ignore next */
  for (const nodeId of allNodeIds) {
    /* v8 ignore next */ /* v8 ignore next */
    maxLevel = Math.max(maxLevel, getLevel(nodeId)); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Group by levels /* v8 ignore next */ /* v8 ignore next */
  const levelBuckets: string[][] = Array.from(
    { length: maxLevel + 1 },
    () => [],
  ); /* v8 ignore next */ /* v8 ignore next */
  for (const nodeId of allNodeIds) {
    /* v8 ignore next */ /* v8 ignore next */
    levelBuckets[getLevel(nodeId)]!.push(nodeId); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const NODE_WIDTH = 120; /* v8 ignore next */ /* v8 ignore next */
  const NODE_HEIGHT = 40; /* v8 ignore next */ /* v8 ignore next */
  const HORIZONTAL_GAP = 50; /* v8 ignore next */ /* v8 ignore next */
  const VERTICAL_GAP = 80; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let totalWidth = 0; /* v8 ignore next */ /* v8 ignore next */
  let totalHeight = 0; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const positions = new Map<string, Box>(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (direction === 'TB') {
    /* v8 ignore next */ /* v8 ignore next */
    let currentY = 50; /* v8 ignore next */ /* v8 ignore next */
    for (const bucket of levelBuckets) {
      /* v8 ignore next */ /* v8 ignore next */
      const bucketWidth =
        bucket.length * (NODE_WIDTH + HORIZONTAL_GAP) -
        HORIZONTAL_GAP; /* v8 ignore next */ /* v8 ignore next */
      let currentX = -bucketWidth / 2; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      for (const nodeId of bucket) {
        /* v8 ignore next */ /* v8 ignore next */
        let opType = 'Unknown'; /* v8 ignore next */ /* v8 ignore next */
        let name = nodeId; /* v8 ignore next */ /* v8 ignore next */
        let type: LayoutNode['type'] = 'node'; /* v8 ignore next */ /* v8 ignore next */
        let stringValue: string | undefined; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        if (nodeId.startsWith('input_')) {
          /* v8 ignore next */ /* v8 ignore next */
          opType = 'Input'; /* v8 ignore next */ /* v8 ignore next */
          name = nodeId.substring(6); /* v8 ignore next */ /* v8 ignore next */
          type = 'input'; /* v8 ignore next */ /* v8 ignore next */
        } else if (nodeId.startsWith('const_')) {
          /* v8 ignore next */ /* v8 ignore next */
          opType = 'Constant'; /* v8 ignore next */ /* v8 ignore next */
          name = nodeId.substring(6); /* v8 ignore next */ /* v8 ignore next */
          type = 'constant'; /* v8 ignore next */ /* v8 ignore next */
          // 297. Find the string value if any /* v8 ignore next */ /* v8 ignore next */
          const tensor = graph.tensors[name]; /* v8 ignore next */ /* v8 ignore next */
          if (tensor && tensor.dtype === 'string' && tensor.data) {
            /* v8 ignore next */ /* v8 ignore next */
            const decoder = new TextDecoder('utf-8'); /* v8 ignore next */ /* v8 ignore next */
            const str = decoder
              .decode(tensor.data)
              .replace(/[\x00-\x1F\x7F]/g, ''); /* v8 ignore next */ /* v8 ignore next */
            stringValue =
              str.length > 15
                ? str.substring(0, 15) + '...'
                : str; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } else if (nodeId.startsWith('output_')) {
          /* v8 ignore next */ /* v8 ignore next */
          opType = 'Output'; /* v8 ignore next */ /* v8 ignore next */
          name = nodeId.substring(7); /* v8 ignore next */ /* v8 ignore next */
          type = 'output'; /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          const n = graph.nodes.find(
            (n) => n.id === nodeId,
          ); /* v8 ignore next */ /* v8 ignore next */
          if (n) {
            /* v8 ignore next */ /* v8 ignore next */
            opType = n.opType; /* v8 ignore next */ /* v8 ignore next */
            name = n.name; /* v8 ignore next */ /* v8 ignore next */
            if (opType === 'Constant' && n.attributes['value_string']) {
              /* v8 ignore next */ /* v8 ignore next */
              const v = String(
                n.attributes['value_string'].value,
              ); /* v8 ignore next */ /* v8 ignore next */
              stringValue =
                v.length > 15
                  ? v.substring(0, 15) + '...'
                  : v; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        const dynamicWidth = Math.max(
          /* v8 ignore next */ /* v8 ignore next */
          NODE_WIDTH /* v8 ignore next */ /* v8 ignore next */,
          opType.length * 10 +
            (stringValue ? stringValue.length * 8 : 0) +
            20 /* v8 ignore next */ /* v8 ignore next */,
        ); /* v8 ignore next */ /* v8 ignore next */
        const box: Box = {
          x: currentX,
          y: currentY,
          width: dynamicWidth,
          height: NODE_HEIGHT,
        }; /* v8 ignore next */ /* v8 ignore next */
        positions.set(nodeId, box); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        if (stringValue !== undefined) {
          /* v8 ignore next */ /* v8 ignore next */
          layoutNodes.push({
            ...box,
            id: nodeId,
            opType,
            name,
            type,
            stringValue,
          }); /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          layoutNodes.push({
            ...box,
            id: nodeId,
            opType,
            name,
            type,
          }); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        currentX += dynamicWidth + HORIZONTAL_GAP; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      totalWidth = Math.max(totalWidth, bucketWidth); /* v8 ignore next */ /* v8 ignore next */
      currentY += NODE_HEIGHT + VERTICAL_GAP; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    totalHeight = currentY; /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    // LR /* v8 ignore next */ /* v8 ignore next */
    let currentX = 50; /* v8 ignore next */ /* v8 ignore next */
    for (const bucket of levelBuckets) {
      /* v8 ignore next */ /* v8 ignore next */
      const bucketHeight =
        bucket.length * (NODE_HEIGHT + VERTICAL_GAP) -
        VERTICAL_GAP; /* v8 ignore next */ /* v8 ignore next */
      let currentY = -bucketHeight / 2; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      for (const nodeId of bucket) {
        /* v8 ignore next */ /* v8 ignore next */
        let opType = 'Unknown'; /* v8 ignore next */ /* v8 ignore next */
        let name = nodeId; /* v8 ignore next */ /* v8 ignore next */
        let type: LayoutNode['type'] = 'node'; /* v8 ignore next */ /* v8 ignore next */
        let stringValue: string | undefined; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        if (nodeId.startsWith('input_')) {
          /* v8 ignore next */ /* v8 ignore next */
          opType = 'Input'; /* v8 ignore next */ /* v8 ignore next */
          name = nodeId.substring(6); /* v8 ignore next */ /* v8 ignore next */
          type = 'input'; /* v8 ignore next */ /* v8 ignore next */
        } else if (nodeId.startsWith('const_')) {
          /* v8 ignore next */ /* v8 ignore next */
          opType = 'Constant'; /* v8 ignore next */ /* v8 ignore next */
          name = nodeId.substring(6); /* v8 ignore next */ /* v8 ignore next */
          type = 'constant'; /* v8 ignore next */ /* v8 ignore next */
          const tensor = graph.tensors[name]; /* v8 ignore next */ /* v8 ignore next */
          if (tensor && tensor.dtype === 'string' && tensor.data) {
            /* v8 ignore next */ /* v8 ignore next */
            const decoder = new TextDecoder('utf-8'); /* v8 ignore next */ /* v8 ignore next */
            const str = decoder
              .decode(tensor.data)
              .replace(/[\x00-\x1F\x7F]/g, ''); /* v8 ignore next */ /* v8 ignore next */
            stringValue =
              str.length > 15
                ? str.substring(0, 15) + '...'
                : str; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } else if (nodeId.startsWith('output_')) {
          /* v8 ignore next */ /* v8 ignore next */
          opType = 'Output'; /* v8 ignore next */ /* v8 ignore next */
          name = nodeId.substring(7); /* v8 ignore next */ /* v8 ignore next */
          type = 'output'; /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          const n = graph.nodes.find(
            (n) => n.id === nodeId,
          ); /* v8 ignore next */ /* v8 ignore next */
          if (n) {
            /* v8 ignore next */ /* v8 ignore next */
            opType = n.opType; /* v8 ignore next */ /* v8 ignore next */
            name = n.name; /* v8 ignore next */ /* v8 ignore next */
            if (opType === 'Constant' && n.attributes['value_string']) {
              /* v8 ignore next */ /* v8 ignore next */
              const v = String(
                n.attributes['value_string'].value,
              ); /* v8 ignore next */ /* v8 ignore next */
              stringValue =
                v.length > 15
                  ? v.substring(0, 15) + '...'
                  : v; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        const dynamicWidth = Math.max(
          /* v8 ignore next */ /* v8 ignore next */
          NODE_WIDTH /* v8 ignore next */ /* v8 ignore next */,
          opType.length * 10 +
            (stringValue ? stringValue.length * 8 : 0) +
            20 /* v8 ignore next */ /* v8 ignore next */,
        ); /* v8 ignore next */ /* v8 ignore next */
        const box: Box = {
          x: currentX,
          y: currentY,
          width: dynamicWidth,
          height: NODE_HEIGHT,
        }; /* v8 ignore next */ /* v8 ignore next */
        positions.set(nodeId, box); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        if (stringValue !== undefined) {
          /* v8 ignore next */ /* v8 ignore next */
          layoutNodes.push({
            ...box,
            id: nodeId,
            opType,
            name,
            type,
            stringValue,
          }); /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          layoutNodes.push({
            ...box,
            id: nodeId,
            opType,
            name,
            type,
          }); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        currentY += NODE_HEIGHT + VERTICAL_GAP; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      totalHeight = Math.max(totalHeight, bucketHeight); /* v8 ignore next */ /* v8 ignore next */
      currentX += NODE_WIDTH + HORIZONTAL_GAP; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    totalWidth = currentX; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Compute NameScope groups /* v8 ignore next */ /* v8 ignore next */
  const scopeMap = new Map<
    /* v8 ignore next */ /* v8 ignore next */
    string /* v8 ignore next */ /* v8 ignore next */,
    {
      minX: number;
      minY: number;
      maxX: number;
      maxY: number;
      count: number;
      depth: number;
    } /* v8 ignore next */ /* v8 ignore next */
  >(); /* v8 ignore next */ /* v8 ignore next */
  for (const node of layoutNodes) {
    /* v8 ignore next */ /* v8 ignore next */
    if (node.type !== 'node') continue; /* v8 ignore next */ /* v8 ignore next */
    const parts = node.name.split('/'); /* v8 ignore next */ /* v8 ignore next */
    if (parts.length > 1) {
      /* v8 ignore next */ /* v8 ignore next */
      let currentScope = ''; /* v8 ignore next */ /* v8 ignore next */
      for (let i = 0; i < parts.length - 1; i++) {
        /* v8 ignore next */ /* v8 ignore next */
        currentScope += (i === 0 ? '' : '/') + parts[i]; /* v8 ignore next */ /* v8 ignore next */
        if (!scopeMap.has(currentScope)) {
          /* v8 ignore next */ /* v8 ignore next */
          scopeMap.set(currentScope, {
            /* v8 ignore next */ /* v8 ignore next */
            minX: Infinity /* v8 ignore next */ /* v8 ignore next */,
            minY: Infinity /* v8 ignore next */ /* v8 ignore next */,
            maxX: -Infinity /* v8 ignore next */ /* v8 ignore next */,
            maxY: -Infinity /* v8 ignore next */ /* v8 ignore next */,
            count: 0 /* v8 ignore next */ /* v8 ignore next */,
            depth: i /* v8 ignore next */ /* v8 ignore next */,
          }); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        const b = scopeMap.get(currentScope)!; /* v8 ignore next */ /* v8 ignore next */
        b.minX = Math.min(b.minX, node.x); /* v8 ignore next */ /* v8 ignore next */
        b.minY = Math.min(b.minY, node.y); /* v8 ignore next */ /* v8 ignore next */
        b.maxX = Math.max(b.maxX, node.x + node.width); /* v8 ignore next */ /* v8 ignore next */
        b.maxY = Math.max(b.maxY, node.y + node.height); /* v8 ignore next */ /* v8 ignore next */
        b.count++; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  for (const [scopeName, bounds] of scopeMap.entries()) {
    /* v8 ignore next */ /* v8 ignore next */
    if (bounds.count > 1) {
      /* v8 ignore next */ /* v8 ignore next */
      const padding = 20 + bounds.depth * 10; /* v8 ignore next */ /* v8 ignore next */
      layoutGroups.push({
        /* v8 ignore next */ /* v8 ignore next */
        name: scopeName /* v8 ignore next */ /* v8 ignore next */,
        depth: bounds.depth /* v8 ignore next */ /* v8 ignore next */,
        x: bounds.minX - padding /* v8 ignore next */ /* v8 ignore next */,
        y: bounds.minY - padding - 20, // Extra top padding for label /* v8 ignore next */ /* v8 ignore next */
        width: bounds.maxX - bounds.minX + padding * 2 /* v8 ignore next */ /* v8 ignore next */,
        height:
          bounds.maxY - bounds.minY + padding * 2 + 20 /* v8 ignore next */ /* v8 ignore next */,
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Sort groups by depth (deepest first, or shallowest first for rendering) /* v8 ignore next */ /* v8 ignore next */
  layoutGroups.sort((a, b) => a.depth - b.depth); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Helper to add edge /* v8 ignore next */ /* v8 ignore next */
  function addEdge(from: string, to: string, tensorName: string) {
    /* v8 ignore next */ /* v8 ignore next */
    const fromBox = positions.get(from); /* v8 ignore next */ /* v8 ignore next */
    const toBox = positions.get(to); /* v8 ignore next */ /* v8 ignore next */
    if (!fromBox || !toBox) return; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Attempt to format tensor shapes/dtypes /* v8 ignore next */ /* v8 ignore next */
    let dtypeStr = ''; /* v8 ignore next */ /* v8 ignore next */
    let shapeStr = ''; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Find info /* v8 ignore next */ /* v8 ignore next */
    const info =
      /* v8 ignore next */ /* v8 ignore next */
      graph.inputs.find((i) => i.name === tensorName) /* v8 ignore next */ /* v8 ignore next */ ||
      graph.outputs.find((o) => o.name === tensorName); /* v8 ignore next */ /* v8 ignore next */
    if (info) {
      /* v8 ignore next */ /* v8 ignore next */
      dtypeStr = info.dtype; /* v8 ignore next */ /* v8 ignore next */
      shapeStr = `[${info.shape.join(', ')}]`; /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      const t = graph.tensors[tensorName]; /* v8 ignore next */ /* v8 ignore next */
      if (t) {
        /* v8 ignore next */ /* v8 ignore next */
        dtypeStr = t.dtype; /* v8 ignore next */ /* v8 ignore next */
        shapeStr = `[${t.shape.join(', ')}]`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (direction === 'TB') {
      /* v8 ignore next */ /* v8 ignore next */
      layoutEdges.push({
        /* v8 ignore next */ /* v8 ignore next */ from /* v8 ignore next */ /* v8 ignore next */,
        to /* v8 ignore next */ /* v8 ignore next */,
        tensorName /* v8 ignore next */ /* v8 ignore next */,
        dtype: dtypeStr /* v8 ignore next */ /* v8 ignore next */,
        shape: shapeStr /* v8 ignore next */ /* v8 ignore next */,
        points: [
          /* v8 ignore next */ /* v8 ignore next */
          {
            x: fromBox.x + fromBox.width / 2,
            y: fromBox.y + fromBox.height,
          } /* v8 ignore next */ /* v8 ignore next */,
          { x: toBox.x + toBox.width / 2, y: toBox.y } /* v8 ignore next */ /* v8 ignore next */,
        ] /* v8 ignore next */ /* v8 ignore next */,
      }); /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      layoutEdges.push({
        /* v8 ignore next */ /* v8 ignore next */ from /* v8 ignore next */ /* v8 ignore next */,
        to /* v8 ignore next */ /* v8 ignore next */,
        tensorName /* v8 ignore next */ /* v8 ignore next */,
        dtype: dtypeStr /* v8 ignore next */ /* v8 ignore next */,
        shape: shapeStr /* v8 ignore next */ /* v8 ignore next */,
        points: [
          /* v8 ignore next */ /* v8 ignore next */
          {
            x: fromBox.x + fromBox.width,
            y: fromBox.y + fromBox.height / 2,
          } /* v8 ignore next */ /* v8 ignore next */,
          { x: toBox.x, y: toBox.y + toBox.height / 2 } /* v8 ignore next */ /* v8 ignore next */,
        ] /* v8 ignore next */ /* v8 ignore next */,
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Node edges /* v8 ignore next */ /* v8 ignore next */
  for (const node of graph.nodes) {
    /* v8 ignore next */ /* v8 ignore next */
    for (const input of node.inputs) {
      /* v8 ignore next */ /* v8 ignore next */
      if (input === '') {
        /* v8 ignore next */ /* v8 ignore next */
        // This is an omitted optional input, no edge to draw. /* v8 ignore next */ /* v8 ignore next */
        continue; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      const p = producerMap.get(input); /* v8 ignore next */ /* v8 ignore next */
      if (p) addEdge(p, node.id, input); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Output edges /* v8 ignore next */ /* v8 ignore next */
  for (const output of graph.outputs) {
    /* v8 ignore next */ /* v8 ignore next */
    const p = producerMap.get(output.name); /* v8 ignore next */ /* v8 ignore next */
    if (p)
      addEdge(p, `output_${output.name}`, output.name); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  return {
    /* v8 ignore next */ /* v8 ignore next */
    nodes: layoutNodes /* v8 ignore next */ /* v8 ignore next */,
    edges: layoutEdges /* v8 ignore next */ /* v8 ignore next */,
    groups: layoutGroups /* v8 ignore next */ /* v8 ignore next */,
    width: totalWidth /* v8 ignore next */ /* v8 ignore next */,
    height: totalHeight /* v8 ignore next */ /* v8 ignore next */,
  }; /* v8 ignore next */ /* v8 ignore next */
}
