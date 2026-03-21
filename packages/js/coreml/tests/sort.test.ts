import { describe, it, expect } from 'vitest';
import { Builder, MILDataType, topologicalSort, Operation, Var } from '../src/index.js';

describe("Topological Sort (Kahn's Algorithm)", () => {
  it('Sorts a simple DAG correctly', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    // y = const()
    // x = const()
    // z = add(x, y)
    // We intentionally add ops out of order
    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1]));
    const y = builder.createVar('y', builder.tensor(MILDataType.FLOAT32, [1]));
    const z = builder.createVar('z', builder.tensor(MILDataType.FLOAT32, [1]));

    const opZ = new Operation('add', { x, y }, [z]);
    const opX = new Operation('const', {}, [x]);
    const opY = new Operation('const', {}, [y]);

    // Random order
    const ops = [opZ, opY, opX];

    const sorted = topologicalSort(ops);
    expect(sorted.length).toBe(3);

    // Z must be after X and Y
    const idxZ = sorted.indexOf(opZ);
    const idxX = sorted.indexOf(opX);
    const idxY = sorted.indexOf(opY);

    expect(idxZ).toBeGreaterThan(idxX);
    expect(idxZ).toBeGreaterThan(idxY);
  });

  it('Handles complex dependencies', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const v1 = builder.createVar('v1', builder.tensor(MILDataType.FLOAT32, [1]));
    const v2 = builder.createVar('v2', builder.tensor(MILDataType.FLOAT32, [1]));
    const v3 = builder.createVar('v3', builder.tensor(MILDataType.FLOAT32, [1]));
    const v4 = builder.createVar('v4', builder.tensor(MILDataType.FLOAT32, [1]));

    const op1 = new Operation('op1', {}, [v1]);
    const op2 = new Operation('op2', { a: v1 }, [v2]);
    const op3 = new Operation('op3', { a: v1 }, [v3]);
    const op4 = new Operation('op4', { a: [v2, v3] }, [v4]); // array input

    const ops = [op4, op3, op2, op1];
    const sorted = topologicalSort(ops);

    expect(sorted.indexOf(op1)).toBeLessThan(sorted.indexOf(op2));
    expect(sorted.indexOf(op1)).toBeLessThan(sorted.indexOf(op3));
    expect(sorted.indexOf(op2)).toBeLessThan(sorted.indexOf(op4));
    expect(sorted.indexOf(op3)).toBeLessThan(sorted.indexOf(op4));
  });

  it('Throws an error if a cycle is detected', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const v1 = builder.createVar('v1', builder.tensor(MILDataType.FLOAT32, [1]));
    const v2 = builder.createVar('v2', builder.tensor(MILDataType.FLOAT32, [1]));

    // cycle: op1 depends on v2, op2 depends on v1
    const op1 = new Operation('op1', { a: v2 }, [v1]);
    const op2 = new Operation('op2', { a: v1 }, [v2]);

    expect(() => topologicalSort([op1, op2])).toThrowError(/Cycle detected/);
  });
});
