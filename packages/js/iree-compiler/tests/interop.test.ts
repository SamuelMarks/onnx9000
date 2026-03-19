import { describe, it, expect } from 'vitest';
import { Region, Operation, Block } from '../src/ir/core.js';
import { MLIRInterop } from '../src/passes/interop.js';

describe('MLIR Interop', () => {
  it('should parse and emit MLIR', () => {
    const interop = new MLIRInterop();
    const mlirText = `
        %1 = web.vm.add.i32 %0, %0
        %2 = web.vm.call @func(%1)
        `;
    const region = interop.parseMLIR(mlirText);

    expect(region.blocks[0].operations.length).toBe(2);

    const output = interop.emitMLIR(region);
    expect(output).toContain('web.vm.add.i32');
    expect(output).toContain('web.vm.call');
  });

  it('should implement all mock endpoints', () => {
    const interop = new MLIRInterop();
    expect(interop.importGoogleIREE).toBeDefined();
    expect(interop.importTF).toBeDefined();
    expect(interop.importPyTorch).toBeDefined();
    expect(interop.mapStableHloToWebMhlo).toBeDefined();
    expect(interop.exportNPMBundle).toBeDefined();
    expect(interop.getSourceMap).toBeDefined();
    expect(interop.registerTransformersBackend).toBeDefined();
  });
});
