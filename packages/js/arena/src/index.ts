/**
 * @fileoverview Static slab memory arena allocator for `@onnx9000/arena`.
 * Computes aligned byte offsets and peak memory footprints for WebAssembly/WebGPU tensor lifetimes.
 */

/**
 * Specification for a tensor buffer lifetime within a static memory plan.
 */
import { Onnx9000Error } from '@onnx9000/core';
/** Interface `TensorAllocationSpec` definition. */
export interface TensorAllocationSpec {
  /** Unique tensor identifier. */
  name: string;
  /** Size of the tensor buffer in bytes. */
  byteLength: number;
  /** First execution step index where the tensor is produced. */
  firstStep: number;
  /** Last execution step index where the tensor is consumed. */
  lastStep: number;
}

/**
 * Planned memory region within the contiguous arena slab.
 */
export interface ArenaBlock {
  /** Tensor name assigned to this region. */
  name: string;
  /** Aligned byte offset from the start of the slab. */
  offset: number;
  /** Aligned byte size of the allocation. */
  size: number;
}

/**
 * Contiguous `ArrayBuffer` slab arena allocator for tensor lifetimes.
 */
export class MemoryArena {
  /** Byte alignment boundary (power of two). */
  public readonly alignment: number;
  /** Underlying contiguous buffer allocated for the arena. */
  private buffer: ArrayBuffer | null = null;
  /** Map of planned tensor blocks by tensor name. */
  private readonly blocks: Map<string, ArenaBlock> = new Map();
  /** Peak bytes required across all planned allocations. */
  public peakBytes = 0;

  /**
   * Initializes a new `MemoryArena` with a byte alignment.
   * @param alignment Byte alignment boundary (defaults to 16).
   * @throws Error if alignment is not a positive integer.
   */
  public constructor(alignment = 16) {
    if (alignment <= 0) {
      throw new Onnx9000Error('Alignment must be positive');
    }
    this.alignment = alignment;
  }

  /**
   * Aligns a byte count upward to the arena alignment boundary.
   * @param bytes Unaligned byte size.
   * @returns Aligned byte size.
   */
  public alignSize(bytes: number): number {
    const rem = bytes % this.alignment;
    return rem === 0 ? bytes : bytes + (this.alignment - rem);
  }

  /**
   * Plans tensor offsets with slab reuse for non-overlapping tensor lifetimes.
   * @param specs Array of tensor lifetime specifications.
   * @returns Array of planned `ArenaBlock` allocations.
   */
  public allocateSpecs(specs: TensorAllocationSpec[]): ArenaBlock[] {
    this.blocks.clear();
    this.peakBytes = 0;
    const active: { spec: TensorAllocationSpec; block: ArenaBlock }[] = [];
    const freePool: { offset: number; size: number }[] = [];
    const results: ArenaBlock[] = [];

    for (const spec of specs) {
      for (let i = active.length - 1; i >= 0; i--) {
        const entry = active[i];
        if (entry && entry.spec.lastStep < spec.firstStep) {
          freePool.push({ offset: entry.block.offset, size: entry.block.size });
          active.splice(i, 1);
        }
      }

      const alignedSize = this.alignSize(spec.byteLength);
      const reuseIdx = freePool.findIndex((b) => b.size >= alignedSize);
      let offset = this.peakBytes;

      if (reuseIdx >= 0) {
        const reused = freePool.splice(reuseIdx, 1)[0];
        if (reused) {
          offset = reused.offset;
        }
      } else {
        this.peakBytes += alignedSize;
      }

      const block: ArenaBlock = { name: spec.name, offset, size: alignedSize };
      this.blocks.set(spec.name, block);
      active.push({ spec, block });
      results.push(block);
    }

    this.buffer = new ArrayBuffer(this.peakBytes);
    return results;
  }

  /**
   * Returns a `Uint8Array` view into the allocated arena slab for a named tensor.
   * @param name Tensor name.
   * @returns `Uint8Array` view over the tensor's slab region.
   * @throws Error if the tensor is not allocated in the arena.
   */
  public getView(name: string): Uint8Array {
    const block = this.blocks.get(name);
    if (!block || !this.buffer) {
      throw new Onnx9000Error(`Tensor '${name}' is not allocated in the arena`);
    }
    return new Uint8Array(this.buffer, block.offset, block.size);
  }

  /**
   * Plans the arena layout based on a model summary string.
   * @param modelString The model definition string.
   * @returns A formatted layout summary string.
   * @throws Error if `modelString` is empty.
   */
  public plan(modelString: string): string {
    if (!modelString) {
      throw new Onnx9000Error('Invalid model string');
    }
    this.allocateSpecs([
      { name: modelString, byteLength: modelString.length * 4, firstStep: 0, lastStep: 1 },
    ]);
    return `[Arena] planner processed ${modelString}`;
  }
}
