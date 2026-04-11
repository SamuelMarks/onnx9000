/* eslint-disable */
import { Block, Region, Operation, Value } from '../ir/core.js';
import * as vm from '../dialects/web/vm.js';

// 79. HAL to VM Lowering Pass
export function lowerHALToVM(region: Region): void {
  const newRegion = new Region();
  const moduleBody = new Block(newRegion);
  newRegion.pushBlock(moduleBody);

  const funcBodyRegion = new Region();
  const funcBodyBlock = new Block(funcBodyRegion);
  funcBodyRegion.pushBlock(funcBodyBlock);

  // 80. Convert HAL command buffer recording into VM API calls
  for (const block of region.blocks) {
    for (const op of block.operations) {
      if (op.opcode === 'web.hal.command_buffer.create') {
        const call = vm.call('hal_cmd_create', [], [{ id: 'i32' }]);
        funcBodyBlock.pushOperation(call);
      } else if (op.opcode === 'web.hal.command_buffer.dispatch') {
        const call = vm.call('hal_cmd_dispatch', op.operands, []);
        funcBodyBlock.pushOperation(call);
      } else if (op.opcode === 'web.hal.buffer.subspan') {
        const call = vm.call('hal_buffer_subspan', op.operands, [{ id: 'i32' }]);
        funcBodyBlock.pushOperation(call);
      }
    }
  }

  funcBodyBlock.pushOperation(vm.returnOp([]));

  // 85. Expose import
  const imp = vm.importOp('hal_cmd_create', 'hal', 'cmd_create');
  moduleBody.pushOperation(imp);

  // Create func
  const func = vm.func('main', [], [], funcBodyRegion);
  moduleBody.pushOperation(func);

  const moduleOp = vm.moduleOp(newRegion);

  // Replace original region contents
  region.blocks.length = 0;
  const finalBlock = new Block(region);
  region.pushBlock(finalBlock);
  finalBlock.pushOperation(moduleOp);
}

// 82, 83. Block Layout Optimization & Register Allocation
export function optimizeAndAllocateRegisters(region: Region): void {
  // Stub for VM block layout & register allocation
  // Would traverse VM blocks and assign flat integer IDs to SSA values
}

// 86, 87, 88, 89. WVM Bytecode Emitter
export class BytecodeEmitter {
  emit(region: Region): Uint8Array {
    // FlatBuffer-like schema and binary opcodes (86, 88)
    const bytecode: number[] = [];

    // Magic header
    bytecode.push(0x57, 0x56, 0x4d, 0x30); // "WVM0"

    for (const block of region.blocks) {
      for (const op of block.operations) {
        if (op.opcode === 'web.vm.module') {
          bytecode.push(0x01); // Module Opcode
        } else if (op.opcode === 'web.vm.func') {
          bytecode.push(0x02); // Func Opcode
        } else if (op.opcode === 'web.vm.call') {
          bytecode.push(0x03); // Call Opcode
        }
        // 89. Encode literal constants would happen here
      }
    }

    return new Uint8Array(bytecode);
  }
}

// 90. CLI Disassembler
export function disassembleWVM(bytecode: Uint8Array): string {
  let out = '';
  if (
    bytecode.length >= 4 &&
    bytecode[0] === 0x57 &&
    bytecode[1] === 0x56 &&
    bytecode[2] === 0x4d &&
    bytecode[3] === 0x30
  ) {
    out += 'WVM0 Header OK\n';
  }

  for (let i = 4; i < bytecode.length; i++) {
    switch (bytecode[i]) {
      case 0x01:
        out += 'Module\n';
        break;
      case 0x02:
        out += 'Func\n';
        break;
      case 0x03:
        out += 'Call\n';
        break;
      default:
        out += `Unknown(0x${bytecode[i]!.toString(16)})\n`;
    }
  }
  return out;
}
