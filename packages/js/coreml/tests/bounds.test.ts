import { describe, it, expect } from 'vitest';
import {
  Builder,
  MILDataType,
  establishMemoryBounds,
  ANELimitsExceededWarning,
} from '../src/index.js';

describe('ANE Memory Boundaries', () => {
  it('Passes for small memory footprints', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    // 1000 float32s = 4000 bytes. well under 2GB.
    const x = builder.createVar('weight', builder.tensor(MILDataType.FLOAT32, [1000]));
    builder.addOp('const', {}, [x]);

    expect(() => establishMemoryBounds(block)).not.toThrow();
  });

  it('Calculates float16 footprint correctly', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('weight', builder.tensor(MILDataType.FLOAT16, [1000]));
    builder.addOp('const', {}, [x]);

    expect(() => establishMemoryBounds(block)).not.toThrow();
  });

  it('Calculates int8 footprint correctly', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('weight', builder.tensor(MILDataType.INT8 as MILDataType, [1000]));
    builder.addOp('const', {}, [x]);

    expect(() => establishMemoryBounds(block)).not.toThrow();
  });

  it('Throws ANELimitsExceededWarning when allocations exceed 2GB', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    // 600,000,000 float32s = 2.4 GB
    const x = builder.createVar(
      'massive_weight',
      builder.tensor(MILDataType.FLOAT32, [600_000_000]),
    );
    builder.addOp('const', {}, [x]);

    expect(() => establishMemoryBounds(block)).toThrowError(ANELimitsExceededWarning);
  });

  it('Ignores dynamic shapes or string symbols safely', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('dynamic_weight', builder.tensor(MILDataType.FLOAT32, ['N', 10]));
    builder.addOp('const', {}, [x]);

    expect(() => establishMemoryBounds(block)).not.toThrow();
  });
});
