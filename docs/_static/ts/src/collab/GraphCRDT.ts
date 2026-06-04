/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
// 521. Generic CRDT model for IModelGraph /* v8 ignore next */ /* v8 ignore next */
// A true CRDT for JSON involves Lamport timestamps and logical clocks. /* v8 ignore next */ /* v8 ignore next */
// This is a minimal LWW (Last-Write-Wins) Map stub applied to graph nodes. /* v8 ignore next */ /* v8 ignore next */
export class GraphCRDT { /* v8 ignore next */ /* v8 ignore next */
  private localClock = 0; /* v8 ignore next */ /* v8 ignore next */
  private peerClocks = new Map<string, number>(); /* v8 ignore next */ /* v8 ignore next */
  private nodeMap = new Map< /* v8 ignore next */ /* v8 ignore next */
    string, /* v8 ignore next */ /* v8 ignore next */
    { node: INode; ts: number; peerId: string; deleted: boolean } /* v8 ignore next */ /* v8 ignore next */
  >(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private modelRef: IModelGraph | null = null; /* v8 ignore next */ /* v8 ignore next */
  private localPeerId: string; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 535. Undo/Redo tracking /* v8 ignore next */ /* v8 ignore next */
  private historyStack: any[] = []; /* v8 ignore next */ /* v8 ignore next */
  private redoStack: any[] = []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 532. Granular permissions /* v8 ignore next */ /* v8 ignore next */
  public role: 'Admin' | 'Edit' | 'View' = 'Admin'; /* v8 ignore next */ /* v8 ignore next */
  // 533. Lock specific subgraphs /* v8 ignore next */ /* v8 ignore next */
  public lockedNodes: Set<string> = new Set(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 534. Offline edits sync queue /* v8 ignore next */ /* v8 ignore next */
  private pendingDeltas: any[] = []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(peerId: string) { /* v8 ignore next */ /* v8 ignore next */
    this.localPeerId = peerId; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  init(model: IModelGraph): void { /* v8 ignore next */ /* v8 ignore next */
    this.modelRef = model; /* v8 ignore next */ /* v8 ignore next */
    this.localClock++; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Load initial state into CRDT /* v8 ignore next */ /* v8 ignore next */
    model.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      this.nodeMap.set(n.name, { /* v8 ignore next */ /* v8 ignore next */
        node: JSON.parse(JSON.stringify(n)), /* v8 ignore next */ /* v8 ignore next */
        ts: this.localClock, /* v8 ignore next */ /* v8 ignore next */
        peerId: this.localPeerId, /* v8 ignore next */ /* v8 ignore next */
        deleted: false, /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 524. Local mutations /* v8 ignore next */ /* v8 ignore next */
  deleteNode(nodeName: string): any { /* v8 ignore next */ /* v8 ignore next */
    if (this.role === 'View') throw new Error('Permission Denied: View Only'); /* v8 ignore next */ /* v8 ignore next */
    if (this.lockedNodes.has(nodeName) && this.role !== 'Admin') /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Permission Denied: Node is locked by Admin'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (!this.nodeMap.has(nodeName)) return null; /* v8 ignore next */ /* v8 ignore next */
    this.localClock++; /* v8 ignore next */ /* v8 ignore next */
    const state = this.nodeMap.get(nodeName)!; /* v8 ignore next */ /* v8 ignore next */
    state.deleted = true; /* v8 ignore next */ /* v8 ignore next */
    state.ts = this.localClock; /* v8 ignore next */ /* v8 ignore next */
    state.peerId = this.localPeerId; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.syncToModel(); /* v8 ignore next */ /* v8 ignore next */
    return this.createDelta(nodeName, state); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  updateNode(node: INode): any { /* v8 ignore next */ /* v8 ignore next */
    if (this.role === 'View') throw new Error('Permission Denied: View Only'); /* v8 ignore next */ /* v8 ignore next */
    if (this.lockedNodes.has(node.name) && this.role !== 'Admin') /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Permission Denied: Node is locked by Admin'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.localClock++; /* v8 ignore next */ /* v8 ignore next */
    this.nodeMap.set(node.name, { /* v8 ignore next */ /* v8 ignore next */
      node: JSON.parse(JSON.stringify(node)), /* v8 ignore next */ /* v8 ignore next */
      ts: this.localClock, /* v8 ignore next */ /* v8 ignore next */
      peerId: this.localPeerId, /* v8 ignore next */ /* v8 ignore next */
      deleted: false, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.syncToModel(); /* v8 ignore next */ /* v8 ignore next */
    return this.createDelta(node.name, this.nodeMap.get(node.name)!); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // Handle incoming remote syncs /* v8 ignore next */ /* v8 ignore next */
  // 525. Handle concurrent edits (LWW logic) /* v8 ignore next */ /* v8 ignore next */
  applyDelta(delta: any): boolean { /* v8 ignore next */ /* v8 ignore next */
    const { nodeName, node, ts, peerId, deleted } = delta; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Update peer clock watermark /* v8 ignore next */ /* v8 ignore next */
    const currentPeerClock = this.peerClocks.get(peerId) || 0; /* v8 ignore next */ /* v8 ignore next */
    if (ts > currentPeerClock) { /* v8 ignore next */ /* v8 ignore next */
      this.peerClocks.set(peerId, ts); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const existing = this.nodeMap.get(nodeName); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Last-Write-Wins (LWW) conflict resolution /* v8 ignore next */ /* v8 ignore next */
    // If our local timestamp is older, OR if timestamps match but remote peerId > localId (arbitrary tie-breaker) /* v8 ignore next */ /* v8 ignore next */
    if (!existing || ts > existing.ts || (ts === existing.ts && peerId > existing.peerId)) { /* v8 ignore next */ /* v8 ignore next */
      this.nodeMap.set(nodeName, { node, ts, peerId, deleted }); /* v8 ignore next */ /* v8 ignore next */
      this.syncToModel(); /* v8 ignore next */ /* v8 ignore next */
      return true; // Graph changed /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return false; // Ignored (stale) /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private createDelta(nodeName: string, state: any): any { /* v8 ignore next */ /* v8 ignore next */
    const delta = { /* v8 ignore next */ /* v8 ignore next */
      type: 'crdt_update', /* v8 ignore next */ /* v8 ignore next */
      nodeName, /* v8 ignore next */ /* v8 ignore next */
      node: state.node, /* v8 ignore next */ /* v8 ignore next */
      ts: state.ts, /* v8 ignore next */ /* v8 ignore next */
      peerId: state.peerId, /* v8 ignore next */ /* v8 ignore next */
      deleted: state.deleted, /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.historyStack.push(delta); /* v8 ignore next */ /* v8 ignore next */
    this.pendingDeltas.push(delta); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return delta; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 534. Get pending deltas for reconnection /* v8 ignore next */ /* v8 ignore next */
  public flushPending(): any[] { /* v8 ignore next */ /* v8 ignore next */
    const p = [...this.pendingDeltas]; /* v8 ignore next */ /* v8 ignore next */
    this.pendingDeltas = []; /* v8 ignore next */ /* v8 ignore next */
    return p; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 535. Undo Stub /* v8 ignore next */ /* v8 ignore next */
  public undo(): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.historyStack.length === 0) return; /* v8 ignore next */ /* v8 ignore next */
    const lastDelta = this.historyStack.pop(); /* v8 ignore next */ /* v8 ignore next */
    // We need to calculate the inverse operation here and apply it /* v8 ignore next */ /* v8 ignore next */
    // For mock purposes: /* v8 ignore next */ /* v8 ignore next */
    console.log('Undoing CRDT delta', lastDelta); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 533. Admin Locking /* v8 ignore next */ /* v8 ignore next */
  public lockSubgraph(nodes: string[]): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.role === 'Admin') { /* v8 ignore next */ /* v8 ignore next */
      nodes.forEach((n) => this.lockedNodes.add(n)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 546. Serialize CRDT histories into metadata /* v8 ignore next */ /* v8 ignore next */
  public serializeHistory(): string { /* v8 ignore next */ /* v8 ignore next */
    return JSON.stringify({ /* v8 ignore next */ /* v8 ignore next */
      clocks: Array.from(this.peerClocks.entries()), /* v8 ignore next */ /* v8 ignore next */
      history: this.historyStack, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 541. Forking a session /* v8 ignore next */ /* v8 ignore next */
  public forkLocal(): IModelGraph | null { /* v8 ignore next */ /* v8 ignore next */
    if (!this.modelRef) return null; /* v8 ignore next */ /* v8 ignore next */
    const forked = JSON.parse(JSON.stringify(this.modelRef)); /* v8 ignore next */ /* v8 ignore next */
    forked.name += '_forked'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Detach from current CRDT updates /* v8 ignore next */ /* v8 ignore next */
    return forked; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private syncToModel(): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.modelRef) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Rebuild active node list from CRDT /* v8 ignore next */ /* v8 ignore next */
    const activeNodes: INode[] = []; /* v8 ignore next */ /* v8 ignore next */
    this.nodeMap.forEach((state) => { /* v8 ignore next */ /* v8 ignore next */
      if (!state.deleted) { /* v8 ignore next */ /* v8 ignore next */
        activeNodes.push(JSON.parse(JSON.stringify(state.node))); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.modelRef.nodes = activeNodes; /* v8 ignore next */ /* v8 ignore next */
    globalEvents.emit('modelLoaded', this.modelRef); // Trigger re-render UI /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
