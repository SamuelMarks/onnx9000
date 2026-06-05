import { Block, Region, Operation, Value } from '../ir/core.js';
import * as hal from '../dialects/web/hal.js';
import * as memref from '../dialects/web/memref.js';

// 60-69. HAL lowering passes
export function lowerLinalgToHAL(region: Region, fallbackBackend: string = 'wasm'): void {
  const memoryArenaSize = 1024 * 1024; // 1MB arena (dummy)
  const deviceType = new hal.DeviceType();
  const bufferType = new hal.BufferType();
  const cmdBufferType = new hal.CommandBufferType();
  const execType = new hal.ExecutableType();

  for (const block of region.blocks) {
    const newOps: Operation[] = [];

    // Setup base device and cmd buffer
    const getDeviceOp = new Operation('web.hal.device.get', [], [deviceType], { default: true });
    newOps.push(getDeviceOp);

    // 64. Static memory planning (dummy arena alloc)
    const arenaAllocOp = new Operation(
      'web.hal.allocator.allocate',
      [getDeviceOp.results[0]!],
      [bufferType],
      { size: memoryArenaSize },
    );
    newOps.push(arenaAllocOp);

    const createCmdBufOp = new Operation(
      'web.hal.command_buffer.create',
      [getDeviceOp.results[0]!],
      [cmdBufferType],
      { mode: 'one_shot' },
    );
    newOps.push(createCmdBufOp);

    const beginCmdBufOp = new Operation(
      'web.hal.command_buffer.begin',
      [createCmdBufOp.results[0]!],
      [],
    );
    newOps.push(beginCmdBufOp);

    let currentOffset = 0;
    const memRefToSubspan = new Map<Value, Value>();

    for (const op of block.operations) {
      if (op.opcode === 'web.memref.alloc') {
        // 65. Emit subspan
        const size = 256; // Compute proper size from shape
        const subspanOp = hal.bufferSubspan(
          arenaAllocOp.results[0]!,
          currentOffset,
          size,
          bufferType,
        );
        newOps.push(subspanOp);
        memRefToSubspan.set(op.results[0]!, subspanOp.results[0]!);
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
          createCmdBufOp.results[0]!,
          createExecOp.results[0]!,
          64,
          64,
          1, // 62. 3D grid
          bindings,
        );
        newOps.push(dispatchOp);
      } else if (op.opcode === 'web.linalg.fill') {
        // 58. Fill buffer
        const fillVal = 0; // dummy
        const target = memRefToSubspan.get(op.operands[1]!) || op.operands[1]!;
        const fillOp = hal.commandBufferFillBuffer(
          createCmdBufOp.results[0]!,
          target,
          0,
          256,
          fillVal,
        );
        newOps.push(fillOp);
      }
    }

    // 66. Command buffer batching / End cmd buffer
    const endCmdBufOp = new Operation(
      'web.hal.command_buffer.end',
      [createCmdBufOp.results[0]!],
      [],
    );
    newOps.push(endCmdBufOp);

    // 67. Synchronization / submission
    const submitOp = new Operation(
      'web.hal.device.queue.submit',
      [getDeviceOp.results[0]!, createCmdBufOp.results[0]!],
      [],
    );
    newOps.push(submitOp);

    const syncOp = new Operation('web.hal.device.queue.wait_idle', [getDeviceOp.results[0]!], []);
    newOps.push(syncOp);

    block.operations.length = 0;
    for (const newOp of newOps) {
      block.pushOperation(newOp);
    }
  }
}
