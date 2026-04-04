import { expect, test } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { Graph } from '../src/ir/graph.js';
import {
  AutoShardingPass,
  SPMDLoweringPass,
  allReduce,
  allGather,
  reduceScatter,
  allToAll,
} from '../src/sharding.js';

test('Sharding Pass', () => {
  const auto = new AutoShardingPass();
  const spmd = new SPMDLoweringPass();
  const g = new Graph();
  expect(auto.apply(g)).toBe(g);
  expect(spmd.apply(g)).toBe(g);
});

test('Sharding Ops', () => {
  const x = new Tensor('x', [1], 1, false, false, new Float32Array());
  x.sharding = [null, 'tp_x'];

  expect(allReduce(x).name).toBe('AllReduce_out');
  expect(allGather(x).name).toBe('AllGather_out');
  expect(reduceScatter(x).name).toBe('ReduceScatter_out');
  expect(allToAll(x).name).toBe('AllToAll_out');
});
