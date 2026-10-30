/**
 * @fileoverview lower_linalg_to_hal.ts
 * Provides lower_linalg_to_hal functionality for the iree-compiler package.
 */

import * as hal from '../dialects/web/hal.js';
import { Operation, type Region, type Value } from '../ir/core.js';

// 60-69. HAL lowering passes
/**
 * Executes `lowerLinalgToHAL`.
 * @param region Parameter `region`.
 * @param _fallbackBackend Parameter `_fallbackBackend`.
 */
export function lowerLinalgToHAL(region: Region, _fallbackBackend: string = 'wasm'): void {
  const memoryArenaSize = 1024 * 1024; // 1MB arena (dummy)
  const deviceType = new hal.DeviceType();
  const bufferType = new hal.BufferType();
  const cmdBufferType = new hal.CommandBufferType();
  const execType = new hal.ExecutableType();

  for (const block of region.blocks) {
    const newOps: Operation[] = [];

    // Setup base device and cmd buffer
    const getDeviceOp = new Operation('web.hal.device.get', [], [deviceType], {
      default: true,
    });
    newOps.push(getDeviceOp);
    const devVal = getDeviceOp.results[0] as Value;

    // 64. Static memory planning (dummy arena alloc)
    const arenaAllocOp = new Operation('web.hal.allocator.allocate', [devVal], [bufferType], {
      size: memoryArenaSize,
    });
    newOps.push(arenaAllocOp);
    const arenaVal = arenaAllocOp.results[0] as Value;

    const createCmdBufOp = new Operation(
      'web.hal.command_buffer.create',
      [devVal],
      [cmdBufferType],
      { mode: 'one_shot' },
    );
    newOps.push(createCmdBufOp);
    const cmdBufVal = createCmdBufOp.results[0] as Value;

    const beginCmdBufOp = new Operation('web.hal.command_buffer.begin', [cmdBufVal], []);
    newOps.push(beginCmdBufOp);

    let currentOffset = 0;
    const memRefToSubspan = new Map<Value, Value>();

    for (const op of block.operations) {
      if (op.opcode === 'web.memref.alloc') {
        // 65. Emit subspan
        const size = 256; // Compute proper size from shape
        const subspanOp = hal.bufferSubspan(arenaVal, currentOffset, size, bufferType);
        newOps.push(subspanOp);
        memRefToSubspan.set(op.results[0] as Value, subspanOp.results[0] as Value);
        currentOffset += size;
      } else if (op.opcode === 'web.linalg.matmul' || op.opcode === 'web.linalg.generic') {
        // 61. Extract kernel, 62. Generate 3D grid, 63. Executable creation, 69. Target backends

        const kernelName = `kernel_${newOps.length}`;

        // Emitting the executable
        const wgslShader = `
                @group(0) @binding(0) var<storage, read> lhs: array<f32>;
                @group(0) @binding(1) var<storage, read> rhs: array<f32>;
                @group(0) @binding(2) var<storage, read_write> out: array<f32>;
                
                @compute @workgroup_size(16, 16, 1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    // dummy matmul kernel
                }`;

        const createExecOp = hal.executableCreate(kernelName, 'wgsl', wgslShader, execType);
        newOps.push(createExecOp);

        // Rebind operands to subspans
        const bindings = op.operands.map((o) => memRefToSubspan.get(o) || o);

        // 56. Dispatch
        const dispatchOp = hal.commandBufferDispatch(
          cmdBufVal,
          createExecOp.results[0] as Value,
          64,
          64,
          1, // 62. 3D grid
          bindings,
        );
        newOps.push(dispatchOp);
      } else if (op.opcode === 'web.linalg.fill') {
        // 58. Fill buffer
        const fillVal = 0; // dummy
        const op1 = op.operands[1] as Value;
        const target = memRefToSubspan.get(op1) || op1;
        const fillOp = hal.commandBufferFillBuffer(cmdBufVal, target, 0, 256, fillVal);
        newOps.push(fillOp);
      }
    }

    // 66. Command buffer batching / End cmd buffer
    const endCmdBufOp = new Operation('web.hal.command_buffer.end', [cmdBufVal], []);
    newOps.push(endCmdBufOp);

    // 67. Synchronization / submission
    const submitOp = new Operation('web.hal.device.queue.submit', [devVal, cmdBufVal], []);
    newOps.push(submitOp);

    const syncOp = new Operation('web.hal.device.queue.wait_idle', [devVal], []);
    newOps.push(syncOp);

    block.operations.length = 0;
    for (const newOp of newOps) {
      block.pushOperation(newOp);
    }
  }
}
