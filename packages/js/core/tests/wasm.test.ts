import { describe, it, expect } from 'vitest';
import * as engine from '../src/wasm/engine.js';

describe('wasm engine', () => {
  it('should initialize', () => {
    expect(engine.init).toBeDefined();
    engine.init();
  });

  it('should execute graph', () => {
    expect(engine.execute_graph).toBeDefined();
    const result = engine.execute_graph(0 as Object, 0 as Object);
    expect(result).toBe(0);
  });
});
