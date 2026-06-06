// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { PipelineHistory } from '../src/core/PipelineHistory.js';

describe('PipelineHistory', () => {
  it('should push, undo and redo', () => {
    const hist = new PipelineHistory();
    hist.push({ sourceFramework: 'keras', targetFramework: 'onnx', activeFile: '' }, 'step1');
    hist.push({ sourceFramework: 'keras', targetFramework: 'c', activeFile: '' }, 'step2');

    expect(hist.getHistory().length).toBe(2);

    const prev = hist.undo();
    expect(prev).toBeDefined();
    expect(hist.getHistory().length).toBe(1);

    const next = hist.redo();
    expect(next).toBeDefined();
    expect(hist.getHistory().length).toBe(2);

    hist.clear();
    expect(hist.getHistory().length).toBe(0);
  });
});
