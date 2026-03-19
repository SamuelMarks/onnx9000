import { Operation, Type, Value, Region } from '../../ir/core.js';

// 51. web.hal.device
export class DeviceType implements Type {
  readonly id = 'hal.device';
}

// 52. web.hal.buffer
export class BufferType implements Type {
  readonly id = 'hal.buffer';
}

// 53. web.hal.buffer_view
export class BufferViewType implements Type {
  readonly id = 'hal.buffer_view';
  constructor(
    public readonly shape: number[],
    public readonly elementType: string,
  ) {}
}

// 54. web.hal.command_buffer
export class CommandBufferType implements Type {
  readonly id = 'hal.command_buffer';
}

// 55. web.hal.executable
export class ExecutableType implements Type {
  readonly id = 'hal.executable';
}

export function executableCreate(
  name: string,
  targetBackend: string,
  shaderCode: string,
  resultType: ExecutableType,
): Operation {
  return new Operation('web.hal.executable.create', [], [resultType], {
    name,
    target_backend: targetBackend,
    shader_code: shaderCode,
  });
}

// 56. web.hal.command_buffer.dispatch
export function commandBufferDispatch(
  cmdBuffer: Value,
  executable: Value,
  workgroupX: number,
  workgroupY: number,
  workgroupZ: number,
  bindings: Value[],
): Operation {
  return new Operation(
    'web.hal.command_buffer.dispatch',
    [cmdBuffer, executable, ...bindings],
    [],
    {
      workgroup_size: [workgroupX, workgroupY, workgroupZ],
    },
  );
}

// 57. web.hal.command_buffer.copy_buffer
export function commandBufferCopyBuffer(
  cmdBuffer: Value,
  sourceBuffer: Value,
  sourceOffset: number,
  targetBuffer: Value,
  targetOffset: number,
  length: number,
): Operation {
  return new Operation(
    'web.hal.command_buffer.copy_buffer',
    [cmdBuffer, sourceBuffer, targetBuffer],
    [],
    {
      source_offset: sourceOffset,
      target_offset: targetOffset,
      length,
    },
  );
}

// 58. web.hal.command_buffer.fill_buffer
export function commandBufferFillBuffer(
  cmdBuffer: Value,
  targetBuffer: Value,
  targetOffset: number,
  length: number,
  pattern: number,
): Operation {
  return new Operation('web.hal.command_buffer.fill_buffer', [cmdBuffer, targetBuffer], [], {
    target_offset: targetOffset,
    length,
    pattern,
  });
}

// 59. web.hal.buffer.subspan
export function bufferSubspan(
  buffer: Value,
  offset: number,
  length: number,
  resultType: BufferType,
): Operation {
  return new Operation('web.hal.buffer.subspan', [buffer], [resultType], {
    offset,
    length,
  });
}

// 68. Dynamic shapes (symbolic vars)
export function dynamicShapeVar(name: string, resultType: Type): Operation {
  return new Operation('web.hal.symbolic_shape_var', [], [resultType], { name });
}

// 70. HAL textual printer
export function printHalGraph(region: Region): string {
  let output = 'HAL Execution Graph:\n';
  for (const block of region.blocks) {
    for (const op of block.operations) {
      output += `  %${op.results.length ? 'result' : '_'} = ${op.opcode}`;
      if (Object.keys(op.attributes).length > 0) {
        output += ` { ${JSON.stringify(op.attributes)} }`;
      }
      output += '\n';
    }
  }
  return output;
}
