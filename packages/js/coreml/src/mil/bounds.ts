/* eslint-disable */
/**
 * ANE boundary definitions and WASM heap enforcement logic.
 * @module
 */
import { Block, Operation, Var } from './ast.js';
import { TensorType, MILDataType } from './types.js';
import { ANELimitsExceededWarning } from './errors.js';

/**
 * Validates the statically resolvable constant memory footprint of the MIL block.
 * Explicitly guards against Chrome/Safari V8 OOM crashes during conversion.
 * @param block - The MIL Block to enforce limits upon.
 * @throws {ANELimitsExceededWarning} If memory footprint exceeds the 2GB threshold.
 */
export function establishMemoryBounds(block: Block): void {
  // 284. Handle extremely large dimensional definitions safely without crashing the V8 engine
  const MAX_HEAP_ALLOCATION_BYTES = 2_000_000_000; // 2GB roughly safe limit for V8
  let totalAllocated = 0;

  for (const op of block.operations) {
    if (op.opType === 'const') {
      const out = op.outputs[0];
      if (out && out.type instanceof TensorType) {
        let size = 1;
        out.type.shape.forEach((d) => {
          if (typeof d === 'number') size *= d;
        });

        // Add primitive memory allocation size
        if (out.type.dataType === MILDataType.FLOAT32 || out.type.dataType === MILDataType.INT32) {
          totalAllocated += size * 4;
        } else if (out.type.dataType === MILDataType.FLOAT16) {
          totalAllocated += size * 2;
        } else {
          totalAllocated += size;
        }

        if (totalAllocated > MAX_HEAP_ALLOCATION_BYTES) {
          throw new ANELimitsExceededWarning(
            'Max V8 Heap allocation exceeded during conversion > 2GB',
          );
        }
      }
    }
  }
}
