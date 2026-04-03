import { describe, it, expect, vi } from 'vitest';
import { KerasWebNNCompiler } from '../src/providers/webnn/webnn_builder.js';

describe('KerasWebNNCompiler', () => {
  it('should build Conv2DBNRelu', () => {
    const builder = {
      conv2d: vi.fn().mockReturnValue('conv_out'),
      batchNormalization: vi.fn().mockReturnValue('bn_out'),
      relu: vi.fn().mockReturnValue('relu_out'),
    };
    const compiler = new KerasWebNNCompiler(builder as any, {} as any);

    const res = compiler.buildConv2DBNRelu(
      'in' as any,
      'w' as any,
      'b' as any,
      'gamma' as any,
      'beta' as any,
      'mean' as any,
      'var' as any,
      { padding: [1, 1], strides: [1, 1], dilations: [2, 2], groups: 2, epsilon: 1e-5 },
    );

    expect(res).toBe('relu_out');
    expect(builder.conv2d).toHaveBeenCalledWith('in', 'w', expect.objectContaining({ bias: 'b' }));
    expect(builder.batchNormalization).toHaveBeenCalledWith(
      'conv_out',
      'mean',
      'var',
      expect.objectContaining({ scale: 'gamma' }),
    );

    // hit default arguments
    compiler.buildConv2DBNRelu(
      'in' as any,
      'w' as any,
      undefined,
      'gamma' as any,
      'beta' as any,
      'mean' as any,
      'var' as any,
      {},
    );
  });

  it('should build SeparableConv2D', () => {
    const builder = {
      conv2d: vi.fn().mockReturnValueOnce('depth_out').mockReturnValueOnce('point_out'),
    };
    const compiler = new KerasWebNNCompiler(builder as any, {} as any);

    const res = compiler.buildSeparableConv2D('in' as any, 'dw' as any, 'pw' as any, 'b' as any, {
      inChannels: 3,
      padding: [1, 1],
      strides: [2, 2],
      dilations: [2, 2],
    });

    expect(res).toBe('point_out');
    expect(builder.conv2d).toHaveBeenCalledTimes(2);

    // hit default arguments
    compiler.buildSeparableConv2D('in' as any, 'dw' as any, 'pw' as any, undefined, {
      inChannels: 3,
    });
  });

  it('should executeAsync', async () => {
    const builder = {
      build: vi.fn().mockResolvedValue('compiled_graph'),
    };
    const context = {
      compute: vi.fn().mockResolvedValue({ outputs: { out: 'buffer' } }),
    };
    const compiler = new KerasWebNNCompiler(builder as any, context as any);

    const outputs = await compiler.executeAsync({ out: 'op' } as any, { in: 'buf' } as any);
    expect(outputs.out).toBe('buffer');
    expect(builder.build).toHaveBeenCalledWith({ out: 'op' });
    expect(context.compute).toHaveBeenCalledWith('compiled_graph', { in: 'buf' }, {});
  });
});
