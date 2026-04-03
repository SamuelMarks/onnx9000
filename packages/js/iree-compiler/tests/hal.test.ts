import { describe, it, expect } from 'vitest';
import { Type, Value, Region, Operation, Block } from '../src/ir/core.js';
import * as hal from '../src/dialects/web/hal.js';

describe('Web HAL Dialect', () => {
  it('should create hal ops', () => {
    const cmdBufferType = new hal.CommandBufferType();
    const bufferType = new hal.BufferType();
    const execType = new hal.ExecutableType();
    const deviceType = new hal.DeviceType();
    const bufferViewType = new hal.BufferViewType([1], 'f32');

    const cmdBuffer = new Value(cmdBufferType, {} as Operation);
    const buffer = new Value(bufferType, {} as Operation);
    const exec = new Value(execType, {} as Operation);

    expect(hal.executableCreate('kernel_0', 'wgsl', 'code', execType).opcode).toBe(
      'web.hal.executable.create',
    );
    expect(hal.commandBufferDispatch(cmdBuffer, exec, 16, 16, 1, []).opcode).toBe(
      'web.hal.command_buffer.dispatch',
    );
    expect(hal.commandBufferCopyBuffer(cmdBuffer, buffer, 0, buffer, 0, 100).opcode).toBe(
      'web.hal.command_buffer.copy_buffer',
    );
    expect(hal.commandBufferFillBuffer(cmdBuffer, buffer, 0, 100, 0).opcode).toBe(
      'web.hal.command_buffer.fill_buffer',
    );
    expect(hal.bufferSubspan(buffer, 0, 100, bufferType).opcode).toBe('web.hal.buffer.subspan');
    expect(hal.dynamicShapeVar('dim0', { id: 'index' }).opcode).toBe('web.hal.symbolic_shape_var');
  });

  it('should print graph', () => {
    const region = new Region();
    const op = hal.executableCreate('kernel_0', 'wgsl', 'code', new hal.ExecutableType());
    const block = new Block(region);
    region.pushBlock(block);
    block.pushOperation(op);

    const text = hal.printHalGraph(region);
    expect(text).toContain('web.hal.executable.create');
    expect(text).toContain('target_backend');
  });

  it('buffer view type', () => {
    const view = new hal.BufferViewType([1], 'f32');
    expect(view.id).toBe('hal.buffer_view');
    expect(view.shape).toEqual([1]);
    expect(view.elementType).toBe('f32');
  });
});

it('should print graph without attributes', () => {
  const region = new Region();
  const op = hal.bufferSubspan(
    new Value(new hal.BufferType(), {} as Operation),
    0,
    100,
    new hal.BufferType(),
  );
  const block = new Block(region);
  region.pushBlock(block);
  block.pushOperation(op);

  const text = hal.printHalGraph(region);
  expect(text).toContain('web.hal.buffer.subspan');
});

it('covers print without results', () => {
  const region = new Region();
  const block = new Block(region);
  region.pushBlock(block);
  block.pushOperation(new Operation('test.no_results', [], []));
  const txt = hal.printHalGraph(region);
  expect(txt).toContain('%_ = test.no_results');
});
