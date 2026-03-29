/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { PipelineHistory } from '../../src/core/PipelineHistory';
import { globalEventBus } from '../../src/core/EventBus';

describe('PipelineHistory', () => {
  let history: PipelineHistory;

  beforeEach(() => {
    history = new PipelineHistory();
    globalEventBus.clearAll();
  });

  it('should push state and increment history length', () => {
    const state = { sourceFramework: 'keras', targetFramework: 'onnx', activeFile: 'a.py' };

    history.push(state, 'Initial Keras Load');

    const histArray = history.getHistory();
    expect(histArray.length).toBe(1);
    expect(histArray[0].state).toEqual(state);
    expect(histArray[0].description).toBe('Initial Keras Load');
  });

  it('should undo safely, reverting to previous state', () => {
    const s1 = { sourceFramework: 'k1', targetFramework: 'o1', activeFile: '1.py' };
    const s2 = { sourceFramework: 'k2', targetFramework: 'o2', activeFile: '2.py' };

    history.push(s1, 'step 1');
    history.push(s2, 'step 2');

    expect(history.getHistory().length).toBe(2);
    expect(history.canUndo()).toBe(true);

    const reverted = history.undo();
    expect(reverted).toEqual(s1);
    expect(history.getHistory().length).toBe(1);
  });

  it('should redo safely after an undo', () => {
    const s1 = { sourceFramework: 'k1', targetFramework: 'o1', activeFile: '1.py' };
    const s2 = { sourceFramework: 'k2', targetFramework: 'o2', activeFile: '2.py' };

    history.push(s1, 'step 1');
    history.push(s2, 'step 2');

    history.undo();
    expect(history.canRedo()).toBe(true);

    const remade = history.redo();
    expect(remade).toEqual(s2);
    expect(history.getHistory().length).toBe(2);
  });

  it('should truncate forward history on new push after undo', () => {
    const s1 = { sourceFramework: 'k1', targetFramework: 'o1', activeFile: '1.py' };
    const s2 = { sourceFramework: 'k2', targetFramework: 'o2', activeFile: '2.py' };
    const s3 = { sourceFramework: 'k3', targetFramework: 'o3', activeFile: '3.py' };

    history.push(s1, 's1');
    history.push(s2, 's2');
    history.undo(); // back to s1

    // Now push s3. s2 should be permanently forgotten.
    history.push(s3, 's3');

    expect(history.canRedo()).toBe(false);
    expect(history.getHistory().length).toBe(2);
    expect(history.getHistory()[1].state).toEqual(s3);
  });

  it('should handle boundaries properly (undo from 0, redo from end)', () => {
    expect(history.canUndo()).toBe(false);
    expect(history.undo()).toBeNull();

    expect(history.canRedo()).toBe(false);
    expect(history.redo()).toBeNull();

    const s1 = { sourceFramework: 'k1', targetFramework: 'o1', activeFile: '1.py' };
    history.push(s1, 's1');

    expect(history.canUndo()).toBe(false);
    expect(history.canRedo()).toBe(false);
  });

  it('should trigger events on global EventBus', () => {
    const addSpy = vi.fn();
    const removeSpy = vi.fn();

    globalEventBus.on('PIPELINE_STEP_ADDED', addSpy);
    globalEventBus.on('PIPELINE_STEP_REMOVED', removeSpy);

    const s1 = { sourceFramework: 'k1', targetFramework: 'o1', activeFile: '1.py' };
    const s2 = { sourceFramework: 'k2', targetFramework: 'o2', activeFile: '2.py' };

    history.push(s1, 's1');
    history.push(s2, 's2');
    expect(addSpy).toHaveBeenCalledTimes(2);

    history.undo();
    expect(removeSpy).toHaveBeenCalledTimes(1);

    history.redo();
    expect(addSpy).toHaveBeenCalledTimes(3);
  });

  it('should ensure immutability', () => {
    const s1 = { sourceFramework: 'k1', targetFramework: 'o1', activeFile: '1.py' };
    history.push(s1, 's1');

    // Mutate original object
    s1.sourceFramework = 'HACKED';

    const hist = history.getHistory();
    expect(hist[0].state.sourceFramework).toBe('k1'); // State untouched
  });

  it('should clear all history', () => {
    const s1 = { sourceFramework: 'k1', targetFramework: 'o1', activeFile: '1.py' };
    history.push(s1, 's1');

    history.clear();
    expect(history.canUndo()).toBe(false);
    expect(history.getHistory().length).toBe(0);
  });
});

it('should handle missing crypto.randomUUID', () => {
  // We are outside the before block maybe? Wait, let's just create a new one to be safe.
  const h2 = new PipelineHistory();

  // Save original
  const origCrypto = global.crypto;

  // Override to delete randomUUID
  Object.defineProperty(global, 'crypto', {
    value: { ...origCrypto, randomUUID: undefined },
    writable: true,
    configurable: true
  });

  const s1 = { sourceFramework: 'k1', targetFramework: 'o1', activeFile: '1.py' };
  h2.push(s1, 'fallback uid test');

  const hist = h2.getHistory();
  expect(hist[0].id).toBeDefined();
  expect(typeof hist[0].id).toBe('string');
  expect(hist[0].id.length).toBeGreaterThan(5);

  // Restore
  Object.defineProperty(global, 'crypto', {
    value: origCrypto,
    writable: true,
    configurable: true
  });
});
