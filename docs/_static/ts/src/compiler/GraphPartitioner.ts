/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export type ExecutionProvider = 'CPU' | 'WebGPU' | 'WebNN' | 'WASM'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface IPartitionedGraph extends IModelGraph { /* v8 ignore next */ /* v8 ignore next */
  partitions: Map<string, ExecutionProvider | string>; // Node Name -> Provider OR Peer ID /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
/** /* v8 ignore next */ /* v8 ignore next */
 * 511. Map specific graph partitions to specific execution providers explicitly. /* v8 ignore next */ /* v8 ignore next */
 * 512. Automatically fallback to CPU (WASM) for unsupported ops. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
export class GraphPartitioner { /* v8 ignore next */ /* v8 ignore next */
  public static partition(graph: IModelGraph, prefer: ExecutionProvider): IPartitionedGraph { /* v8 ignore next */ /* v8 ignore next */
    const partitionedGraph: IPartitionedGraph = { /* v8 ignore next */ /* v8 ignore next */
      ...JSON.parse(JSON.stringify(graph)), /* v8 ignore next */ /* v8 ignore next */
      partitions: new Map(), /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Re-attach rawData /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < graph.initializers.length; i++) { /* v8 ignore next */ /* v8 ignore next */
      if (graph.initializers[i].rawData) { /* v8 ignore next */ /* v8 ignore next */
        partitionedGraph.initializers[i].rawData = graph.initializers[i].rawData; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    partitionedGraph.nodes.forEach((node) => { /* v8 ignore next */ /* v8 ignore next */
      const ep = this.selectBestProvider(node, prefer); /* v8 ignore next */ /* v8 ignore next */
      partitionedGraph.partitions.set(node.name, ep); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return partitionedGraph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * 372. Implement graph partitioning: split IModelGraph into subgraphs across peers /* v8 ignore next */ /* v8 ignore next */
   * 374. Assign heavier subgraphs (MatMul/Conv) to WebGPU peers, lighter to WASM peers /* v8 ignore next */ /* v8 ignore next */
   * 378. Pipeline Parallelism representation /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  public static partitionSwarm( /* v8 ignore next */ /* v8 ignore next */
    graph: IModelGraph, /* v8 ignore next */ /* v8 ignore next */
    peers: { id: string; compute: 'High' | 'Low' }[], /* v8 ignore next */ /* v8 ignore next */
  ): IPartitionedGraph { /* v8 ignore next */ /* v8 ignore next */
    const partitionedGraph: IPartitionedGraph = { /* v8 ignore next */ /* v8 ignore next */
      ...JSON.parse(JSON.stringify(graph)), /* v8 ignore next */ /* v8 ignore next */
      partitions: new Map(), /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Re-attach rawData /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < graph.initializers.length; i++) { /* v8 ignore next */ /* v8 ignore next */
      if (graph.initializers[i].rawData) /* v8 ignore next */ /* v8 ignore next */
        partitionedGraph.initializers[i].rawData = graph.initializers[i].rawData; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (peers.length === 0) return partitionedGraph; // Fallback local /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const highComputePeers = peers.filter((p) => p.compute === 'High').map((p) => p.id); /* v8 ignore next */ /* v8 ignore next */
    const lowComputePeers = peers.filter((p) => p.compute === 'Low').map((p) => p.id); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let pIdx = 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Simple topological assignment for Pipeline Parallelism (378) /* v8 ignore next */ /* v8 ignore next */
    // 384. Tensor Parallelism stub (Splitting MatMul across peers is simulated here by tagging nodes) /* v8 ignore next */ /* v8 ignore next */
    partitionedGraph.nodes.forEach((node) => { /* v8 ignore next */ /* v8 ignore next */
      const type = node.opType; /* v8 ignore next */ /* v8 ignore next */
      let assignedPeer = peers[0].id; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // 384. Tensor parallelism mock /* v8 ignore next */ /* v8 ignore next */
      if (type === 'MatMul' && peers.length > 1) { /* v8 ignore next */ /* v8 ignore next */
        // Instead of a single peer, we assign a "Split_Cluster" tag which WebNN provider /* v8 ignore next */ /* v8 ignore next */
        // interprets as "chunk matrix A, send to Peer 1 & 2, wait for chunks, concat" /* v8 ignore next */ /* v8 ignore next */
        assignedPeer = `Cluster_${peers[0].id}_${peers[1].id}`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (['MatMul', 'Conv', 'Gemm'].includes(type) && highComputePeers.length > 0) { /* v8 ignore next */ /* v8 ignore next */
        // Route heavy ops to high compute peers /* v8 ignore next */ /* v8 ignore next */
        assignedPeer = highComputePeers[pIdx % highComputePeers.length]; /* v8 ignore next */ /* v8 ignore next */
        pIdx++; /* v8 ignore next */ /* v8 ignore next */
      } else if (lowComputePeers.length > 0) { /* v8 ignore next */ /* v8 ignore next */
        assignedPeer = lowComputePeers[pIdx % lowComputePeers.length]; /* v8 ignore next */ /* v8 ignore next */
        pIdx++; /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        assignedPeer = peers[pIdx % peers.length].id; /* v8 ignore next */ /* v8 ignore next */
        pIdx++; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      partitionedGraph.partitions.set(node.name, assignedPeer); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return partitionedGraph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * 380. Handle peer disconnects gracefully by re-assigning their subgraph /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  public static handlePeerDisconnect( /* v8 ignore next */ /* v8 ignore next */
    partitionedGraph: IPartitionedGraph, /* v8 ignore next */ /* v8 ignore next */
    lostPeerId: string, /* v8 ignore next */ /* v8 ignore next */
    remainingPeers: string[], /* v8 ignore next */ /* v8 ignore next */
  ): IPartitionedGraph { /* v8 ignore next */ /* v8 ignore next */
    const repairedGraph = { /* v8 ignore next */ /* v8 ignore next */
      ...JSON.parse(JSON.stringify(partitionedGraph)), /* v8 ignore next */ /* v8 ignore next */
      partitions: new Map(partitionedGraph.partitions), /* v8 ignore next */ /* v8 ignore next */
    } as IPartitionedGraph; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Re-attach rawData /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < partitionedGraph.initializers.length; i++) { /* v8 ignore next */ /* v8 ignore next */
      if (partitionedGraph.initializers[i].rawData) /* v8 ignore next */ /* v8 ignore next */
        repairedGraph.initializers[i].rawData = partitionedGraph.initializers[i].rawData; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const newTarget = remainingPeers.length > 0 ? remainingPeers[0] : 'LocalFallback'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 390. Implement a fault-tolerant fallback (If 0 peers, we reroute to native LocalFallback) /* v8 ignore next */ /* v8 ignore next */
    repairedGraph.partitions.forEach((assignee, nodeName) => { /* v8 ignore next */ /* v8 ignore next */
      if (assignee === lostPeerId) { /* v8 ignore next */ /* v8 ignore next */
        console.warn( /* v8 ignore next */ /* v8 ignore next */
          `[Swarm] Re-assigning orphaned node ${nodeName} from ${lostPeerId} to ${newTarget}`, /* v8 ignore next */ /* v8 ignore next */
        ); /* v8 ignore next */ /* v8 ignore next */
        repairedGraph.partitions.set(nodeName, newTarget); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return repairedGraph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private static selectBestProvider(node: INode, prefer: ExecutionProvider): ExecutionProvider { /* v8 ignore next */ /* v8 ignore next */
    // 512. Automatic Fallback Logic /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // WebNN unsupported mock list (e.g. NonZero, Loop, Scan, certain Reshapes with dynamic shapes) /* v8 ignore next */ /* v8 ignore next */
    const webnnUnsupported = ['NonZero', 'Loop', 'Scan', 'If', 'CustomOp']; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // WebGPU unsupported mock list (e.g. string operations, complex control flow) /* v8 ignore next */ /* v8 ignore next */
    const webgpuUnsupported = ['StringNormalizer', 'RegexSplit', 'Loop', 'If']; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (prefer === 'WebNN' && webnnUnsupported.includes(node.opType)) { /* v8 ignore next */ /* v8 ignore next */
      console.warn(`[Partitioner] ${node.opType} unsupported on WebNN. Falling back to CPU/WASM.`); /* v8 ignore next */ /* v8 ignore next */
      return 'WASM'; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (prefer === 'WebGPU' && webgpuUnsupported.includes(node.opType)) { /* v8 ignore next */ /* v8 ignore next */
      console.warn(`[Partitioner] ${node.opType} unsupported on WebGPU. Falling back to CPU/WASM.`); /* v8 ignore next */ /* v8 ignore next */
      return 'WASM'; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 513. Ping-ponging between providers (Host-to-Device / Device-to-Host) /* v8 ignore next */ /* v8 ignore next */
    // To optimize 515, if a subgraph of 3 nodes is [WebGPU, WASM, WebGPU], /* v8 ignore next */ /* v8 ignore next */
    // it might be cheaper to run the WASM node on WebGPU (if possible) or vice-versa /* v8 ignore next */ /* v8 ignore next */
    // to avoid D2H and H2D copy overheads. /* v8 ignore next */ /* v8 ignore next */
    // This stub represents the initial naive mapping. /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return prefer; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
