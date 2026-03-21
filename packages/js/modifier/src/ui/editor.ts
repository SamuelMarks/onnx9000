import { Graph, Node } from '@onnx9000/core';
import { GraphMutator } from '../GraphMutator.js';

/**
 * Event details for the GraphEditor
 */
export interface SelectionEvent {
  type: 'node' | 'edge';
  id: string; // Node ID or Edge Name
}

/**
 * Handles core UI state for the Modifier Graph View
 */
export class GraphEditor {
  graph: Graph;
  mutator: GraphMutator;

  // 52. Single select, 53. Multi select
  selectedNodeIds: Set<string> = new Set();
  selectedEdges: Set<string> = new Set();

  onSelectionChange: (selection: SelectionEvent[]) => void = () => {};

  constructor(graph: Graph, mutator: GraphMutator) {
    this.graph = graph;
    this.mutator = mutator;
  }

  // 52. Implement node selection logic (single click)
  selectNode(nodeId: string, multiSelect: boolean = false) {
    if (!multiSelect) {
      this.selectedNodeIds.clear();
      this.selectedEdges.clear();
    }
    this.selectedNodeIds.add(nodeId);
    this._notifySelection();
  }

  selectEdge(edgeName: string, multiSelect: boolean = false) {
    if (!multiSelect) {
      this.selectedNodeIds.clear();
      this.selectedEdges.clear();
    }
    this.selectedEdges.add(edgeName);
    this._notifySelection();
  }

  clearSelection() {
    this.selectedNodeIds.clear();
    this.selectedEdges.clear();
    this._notifySelection();
  }

  private _notifySelection() {
    const events: SelectionEvent[] = [];
    for (const id of this.selectedNodeIds) events.push({ type: 'node', id });
    for (const name of this.selectedEdges) events.push({ type: 'edge', id: name });
    this.onSelectionChange(events);
  }

  // 59. Implement a "Delete" button (and mapping)
  deleteSelection() {
    for (const nodeId of this.selectedNodeIds) {
      // clone id since removeNode might alter node names under the hood
      this.mutator.removeNode(nodeId, false);
    }

    // 62. Edge deletion (requires injecting a break in the graph)
    // Simply rename the target port or remove the edge from the consumer
    for (const edgeName of this.selectedEdges) {
      for (const node of this.graph.nodes) {
        const idx = node.inputs.indexOf(edgeName);
        if (idx !== -1) {
          // Break connection by clearing the input
          this.mutator.execute({
            undo: () => {
              node.inputs[idx] = edgeName;
            },
            redo: () => {
              node.inputs[idx] = '';
            },
          });
        }
      }
    }

    this.clearSelection();
  }

  // 60. Context Menu -> Disconnect
  disconnectNode(nodeId: string) {
    const node = this.graph.nodes.find((n) => n.id === nodeId);
    if (!node) return;

    // Disconnect inputs
    const oldInputs = [...node.inputs];
    this.mutator.execute({
      undo: () => {
        node.inputs = [...oldInputs];
      },
      redo: () => {
        node.inputs = node.inputs.map(() => '');
      },
    });
  }

  // 179. Handle copy/pasting subgraphs entirely
  // 259. Manage complex naming collisions when duplicating nodes
  duplicateSubgraph(nodeIds: string[]) {
    const nodesToDuplicate = this.graph.nodes.filter((n) => nodeIds.includes(n.id));
    if (nodesToDuplicate.length === 0) return;

    // Track internal edges to rewire them appropriately
    const internalEdges = new Set<string>();
    for (const node of nodesToDuplicate) {
      for (const out of node.outputs) internalEdges.add(out);
    }

    const renameMap = new Map<string, string>();
    const dupNodes: Node[] = [];
    const originalNodesStr = JSON.stringify(this.graph.nodes);

    for (const node of nodesToDuplicate) {
      const newName = `${node.name || node.opType}_dup_${Math.floor(Math.random() * 10000)}`;

      const newOutputs = node.outputs.map((o) => {
        const renamed = `${o}_dup_${Math.floor(Math.random() * 10000)}`;
        renameMap.set(o, renamed);
        return renamed;
      });

      const newNode = new Node(
        node.opType,
        [...node.inputs], // we will rewire inputs in a second pass
        newOutputs,
        JSON.parse(JSON.stringify(node.attributes)),
        newName,
      );
      dupNodes.push(newNode);
    }

    // Rewire internal edges
    for (const node of dupNodes) {
      node.inputs = node.inputs.map((inp) => (renameMap.has(inp) ? renameMap.get(inp)! : inp));
    }

    this.mutator.execute({
      undo: () => {
        this.graph.nodes = JSON.parse(originalNodesStr);
      },
      redo: () => {
        for (const node of dupNodes) {
          this.graph.addNode(node);
        }
      },
    });
  }

  // 60. Context Menu -> Duplicate

  duplicateNode(nodeId: string) {
    const node = this.graph.nodes.find((n) => n.id === nodeId);
    if (!node) return;

    // Create unique output names
    const newOutputs = node.outputs.map((o) => `${o}_dup`);
    this.mutator.addNode(
      node.opType,
      [...node.inputs],
      newOutputs,
      { ...node.attributes },
      `${node.name || node.opType}_dup`,
    );
  }

  // 61. Edge dragging
  connectPorts(sourceEdgeName: string, targetNodeId: string, targetInputIndex: number) {
    const targetNode = this.graph.nodes.find((n) => n.id === targetNodeId);
    if (!targetNode) return;

    const oldInput = targetNode.inputs[targetInputIndex];

    this.mutator.execute({
      undo: () => {
        targetNode.inputs[targetInputIndex] = oldInput || '';
      },
      redo: () => {
        targetNode.inputs[targetInputIndex] = sourceEdgeName;
      },
    });
  }

  // 202. Align Utilities
  alignNodes(direction: 'Left' | 'Right' | 'Center') {
    // The visual coordinates are technically controlled by the DagreLayoutEngine.
    // However, if we support manual coordinates, we would update them here.
    // For now, we update an attribute on the nodes that could hint the renderer.
    for (const nodeId of this.selectedNodeIds) {
      const node = this.graph.nodes.find((n) => n.id === nodeId);
      if (node) {
        this.mutator.setNodeAttribute(node.id, 'alignment', direction, 'STRING');
      }
    }
  }

  // 201. Grid Snapping
  snapToGrid(gridSize: number = 20) {
    // Updates coordinate attributes if they exist
    for (const nodeId of this.selectedNodeIds) {
      const node = this.graph.nodes.find((n) => n.id === nodeId);
      if (node) {
        // mock snapping by setting snapped attribute
        this.mutator.setNodeAttribute(node.id, 'snapped_grid', gridSize, 'INT');
      }
    }
  }

  // 265. Pin Nodes
  pinNodes() {
    for (const nodeId of this.selectedNodeIds) {
      const node = this.graph.nodes.find((n) => n.id === nodeId);
      if (node) {
        this.mutator.setNodeAttribute(node.id, 'pinned', 1, 'INT');
      }
    }
  }

  unpinNodes() {
    for (const nodeId of this.selectedNodeIds) {
      const node = this.graph.nodes.find((n) => n.id === nodeId);
      if (node) {
        this.mutator.removeNodeAttribute(node.id, 'pinned');
      }
    }
  }
}
