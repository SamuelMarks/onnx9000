import { describe, it, expect } from 'vitest';
import { Region, Operation, Block, Value } from '../src/ir/core.js';
import { TensorType } from '../src/dialects/web/tensor.js';
import { MLIRInterop } from '../src/passes/interop.js';

describe('MLIR Interop', () => {
  it('should parse and emit MLIR', () => {
    const interop = new MLIRInterop();
    const mlirText = `
        %1 = web.vm.add.i32 %0, %0
        %2 = web.vm.call @func(%1)
        `;
    const region = interop.parseMLIR(mlirText);

    expect(region.blocks[0]!.operations.length).toBe(2);

    const output = interop.emitMLIR(region);
    expect(output).toContain('web.vm.add.i32');
    expect(output).toContain('web.vm.call');
  });

  it('covers all branch points in formatType and emitMLIR', () => {
    const interop = new MLIRInterop();
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    // Add args to block
    block.args.push(new Value({ id: 'none' } as any));
    block.args.push(new Value(new TensorType([2], 'float32')));
    block.args.push(new Value(new TensorType([2], 'int64')));
    block.args.push(new Value(new TensorType([-1], 'int32')));
    block.args.push(new Value(new TensorType([], 'bool')));

    // Op with 0 operands, 0 results, 0 attrs
    block.pushOperation(new Operation('test.op0', [], [], {}));

    // Op with 1 result, >0 operands
    const v1 = new Value(new TensorType([], 'bool'));
    block.pushOperation(new Operation('test.op1', [block.args[0]!], [v1], {}));

    // Op with >1 results, >0 attrs (string, array, boolean)
    const v2 = new Value(new TensorType([1], 'float32'));
    const v3 = new Value(new TensorType([1], 'float32'));
    block.pushOperation(
      new Operation('test.op2', [], [v2, v3], {
        str: 'hello',
        arr: [1, 2],
        b: true,
      }),
    );

    const text = interop.emitMLIR(region);
    expect(text).toContain('test.op0');
    expect(text).toContain('test.op1');
    expect(text).toContain('test.op2');
  });

  it('should call all mock endpoints', () => {
    const interop = new MLIRInterop();
    expect(interop.importGoogleIREE('')).toBeInstanceOf(Region);
    expect(interop.importTF('')).toBeInstanceOf(Region);
    expect(interop.importPyTorch('')).toBeInstanceOf(Region);
    const r = new Region();
    expect(interop.mapStableHloToWebMhlo(r)).toBe(r);
    interop.exportNPMBundle('code', 'dir');
    expect(interop.getSourceMap()).toEqual({ 'web.vm.add.i32': 'onnx.node_54' });
    interop.registerTransformersBackend();
  });
});
