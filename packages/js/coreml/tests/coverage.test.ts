import { describe, expect, it } from 'vitest';
import { ONNXToMILConverter } from '../src/converter';
import { Model } from '../src/schema';
import { MLPackageBuilder } from '../src/mlpackage';
import { Block, Var, Operation } from '../src/mil/ast';
import { MILDataType, TensorType, ScalarType } from '../src/mil/types';
import { validateBlock } from '../src/mil/validator';
import { inferShapes } from '../src/mil/rewriter';
import { commonSubexpressionElimination } from '../src/mil/passes';
import { optimizeForANE } from '../src/mil/ane_passes';
import { applyCompression } from '../src/mil/compression';

describe('Coverage tests', () => {
  it('covers validator missing output', () => {
    const block = new Block('test_block');
    const outVar = new Var('missing_out', new ScalarType(MILDataType.FLOAT32));
    block.outputs.push(outVar);
    expect(() => validateBlock(block)).toThrow(/is not produced within block/);
  });

  it('covers rewriter setting shape for general ops', () => {
    const block = new Block('test_block');
    const outVar = new Var('out1', new TensorType(MILDataType.FLOAT32, [1, 2, 3]));
    const op = new Operation('add', {}, [outVar], {});
    block.addOperation(op);
    inferShapes(block);
    expect(outVar.type instanceof TensorType && outVar.type.shape).toEqual([1, 2, 3]);
  });

  it('covers common subexpression elimination array replacement', () => {
    const block = new Block('test_block');
    const inVar = new Var('in', new TensorType(MILDataType.FLOAT32, [1]));
    const out1 = new Var('out1', new TensorType(MILDataType.FLOAT32, [1]));
    const out2 = new Var('out2', new TensorType(MILDataType.FLOAT32, [1]));
    const out3 = new Var('out3', new TensorType(MILDataType.FLOAT32, [1]));

    const op1 = new Operation('relu', { x: inVar }, [out1], {});
    const op2 = new Operation('relu', { x: inVar }, [out2], {});
    const op3 = new Operation('concat', { x: [out1, out2] }, [out3], {});

    block.addOperation(op1);
    block.addOperation(op2);
    block.addOperation(op3);

    commonSubexpressionElimination(block);

    const concatOp = block.operations.find((o) => o.opType === 'concat');
    expect((concatOp!.inputs.x as Var[])[1].name).toBe('out1');
  });

  it('covers ane redundant cast pass array branch', () => {
    const block = new Block('test_block');
    const inVar = new Var('in', new TensorType(MILDataType.FLOAT16, [1]));
    const outVar = new Var('out', new TensorType(MILDataType.FLOAT16, [1]));
    const op = new Operation('cast', { x: [inVar] }, [outVar], { dtype: 'fp16' });
    block.addOperation(op);
    optimizeForANE(block);
    expect(op.opType).toBe('identity');
  });

  it('covers converter attributes and graphs', () => {
    const mockGraph: any = {
      inputs: [
        { name: 'cond', dtype: 'bool', shape: [1] },
        { name: 'a', dtype: 'float32', shape: [1] },
        { name: 'b', dtype: 'float32', shape: [1] },
      ],
      outputs: [],
      initializers: [],
      initializers: [],
      nodes: [
        {
          opType: 'If',
          inputs: ['cond'],
          outputs: ['out'],
          attributes: {
            then_branch: {
              type: 'GRAPH',
              value: {
                initializers: [],
                nodes: [],
                inputs: [],
                outputs: [],
                initializers: [],
              },
            },
          },
        },
        {
          opType: 'Add',
          inputs: ['a', 'b'],
          outputs: ['out2'],
          attributes: {
            axis: { type: 'INT', value: 1 },
          },
        },
      ],
    };
    const converter = new ONNXToMILConverter(mockGraph);
    expect(() => converter.convert()).not.toThrow();
  });

  it('covers mlpackage metadata author and description', () => {
    const onnxModel: any = {
      irVersion: 8n,
      opsetImport: [],
      producerName: 'test',
      description: {
        input: [],
        output: [],
        metadata: {
          author: 'Test Author',
          shortDescription: 'Test Desc',
        },
      },
      graph: {
        node: [],
        input: [],
        output: [],
        initializer: [],
        sparseInitializer: [],
        valueInfo: [],
      },
    };
    const builder = new MLPackageBuilder(onnxModel, new Uint8Array(0), {
      outputDir: 'test_out',
      stateful: true,
      visionFrameworkDescription: '',
    });
    const map = builder.buildDirectoryStructure();
    const manifestBytes = map.get('Manifest.json');
    const manifestStr = new TextDecoder().decode(manifestBytes!);
    expect(manifestStr).toContain('Test Author');
    expect(manifestStr).toContain('Test Desc');
  });

  it('covers compression mixed mode, kv_cache, and reduction percentage', () => {
    const block = new Block('test_block');
    const outVar1 = new Var('out1', new TensorType(MILDataType.FLOAT32, [1000]));
    const op1 = new Operation('const', {}, [outVar1], {});
    const op_rs = new Operation('read_state', {}, [outVar1], {});
    block.addOperation(op_rs);
    const outVar2 = new Var('out2', new TensorType(MILDataType.FLOAT32, [1000]));
    const op2 = new Operation('add', {}, [outVar2], {});
    const outVar3 = new Var('out3', new TensorType(MILDataType.FLOAT32, []));
    const op3 = new Operation('add', {}, [outVar3], {});

    block.addOperation(op1);
    block.addOperation(op2);
    block.addOperation(op3);

    const report1 = applyCompression(block, {
      mode: 'mixed',
      mixedPrecisionConfig: { out1: 'w4a16' },
      gatherStatistics: true,
      kvCacheQuantization: true,
      reportReduction: true,
    });

    expect(report1!.reductionPercentage).toBeGreaterThan(0);
    expect(op_rs.attributes['kv_cache_quantized']).toBe('int4');

    const block2 = new Block('test_block2');
    const report2 = applyCompression(block2, { reportReduction: true });
    expect(report2!.reductionPercentage).toBe(0);
  });
});
