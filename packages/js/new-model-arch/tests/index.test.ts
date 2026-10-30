import { describe, expect, it } from 'vitest';
import { NewModelArch } from '../src/index.js';

describe('NewModelArch', () => {
  it('should build architecture specifications and process requests', () => {
    const arch = new NewModelArch();
    expect(arch.process('mamba-130m')).toBe('New Model Arch processed mamba-130m');

    const llama = arch.buildSpec('llama', 4096, 32);
    expect(llama.hiddenSize).toBe(4096);
    expect(llama.numBlocks).toBe(32);
    expect(llama.blockOps).toContain('GroupQueryAttention');
  });
});
