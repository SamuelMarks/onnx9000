/**
 * @fileoverview hal.ts
 * Provides hal functionality for the iree-compiler package.
 */
import { Operation, type Region, type Type, type Value } from '../../ir/core.js';

// 51. web.hal.device
/** Represents the `DeviceType` class. */
export class DeviceType implements Type {
  /** Property `id` of `DeviceType`. */
  readonly id = 'hal.device';
}

// 52. web.hal.buffer
/** Represents the `BufferType` class. */
export class BufferType implements Type {
  /** Property `id` of `BufferType`. */
  readonly id = 'hal.buffer';
}

// 53. web.hal.buffer_view
/** Represents the `BufferViewType` class. */
export class BufferViewType implements Type {
  /** Property `id` of `BufferViewType`. */
  readonly id = 'hal.buffer_view';
  /**
   * Initializes a new `BufferViewType` instance.
   * @param shape Parameter `shape`.
   * @param elementType Parameter `elementType`.
   */
  constructor(
    public readonly shape: number[],
    public readonly elementType: string,
  ) {}
}

// 54. web.hal.command_buffer
/** Represents the `CommandBufferType` class. */
export class CommandBufferType implements Type {
  /** Property `id` of `CommandBufferType`. */
  readonly id = 'hal.command_buffer';
}

// 55. web.hal.executable
/** Represents the `ExecutableType` class. */
export class ExecutableType implements Type {
  /** Property `id` of `ExecutableType`. */
  readonly id = 'hal.executable';
}

/**
 * Executes `executableCreate`.
 * @param name Parameter `name`.
 * @param targetBackend Parameter `targetBackend`.
 * @param shaderCode Parameter `shaderCode`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `executableCreate`.
 */
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
/**
 * Executes `commandBufferDispatch`.
 * @param cmdBuffer Parameter `cmdBuffer`.
 * @param executable Parameter `executable`.
 * @param workgroupX Parameter `workgroupX`.
 * @param workgroupY Parameter `workgroupY`.
 * @param workgroupZ Parameter `workgroupZ`.
 * @param bindings Parameter `bindings`.
 * @returns Result of `commandBufferDispatch`.
 */
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
/**
 * Executes `commandBufferCopyBuffer`.
 * @param cmdBuffer Parameter `cmdBuffer`.
 * @param sourceBuffer Parameter `sourceBuffer`.
 * @param sourceOffset Parameter `sourceOffset`.
 * @param targetBuffer Parameter `targetBuffer`.
 * @param targetOffset Parameter `targetOffset`.
 * @param length Parameter `length`.
 * @returns Result of `commandBufferCopyBuffer`.
 */
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
/**
 * Executes `commandBufferFillBuffer`.
 * @param cmdBuffer Parameter `cmdBuffer`.
 * @param targetBuffer Parameter `targetBuffer`.
 * @param targetOffset Parameter `targetOffset`.
 * @param length Parameter `length`.
 * @param pattern Parameter `pattern`.
 * @returns Result of `commandBufferFillBuffer`.
 */
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
/**
 * Executes `bufferSubspan`.
 * @param buffer Parameter `buffer`.
 * @param offset Parameter `offset`.
 * @param length Parameter `length`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `bufferSubspan`.
 */
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
/**
 * Executes `dynamicShapeVar`.
 * @param name Parameter `name`.
 * @param resultType Parameter `resultType`.
 * @returns Result of `dynamicShapeVar`.
 */
export function dynamicShapeVar(name: string, resultType: Type): Operation {
  return new Operation('web.hal.symbolic_shape_var', [], [resultType], {
    name,
  });
}

// 70. HAL textual printer
/**
 * Executes `printHalGraph`.
 * @param region Parameter `region`.
 * @returns Result of `printHalGraph`.
 */
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
