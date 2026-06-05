import { describe, it, expect } from 'vitest';
import {
  registerCustomKerasLayer,
  getCustomKerasLayerEmitter,
} from '../../src/keras/plugin-registry.js';

describe('plugin-registry', () => {
  it('should register plugin', () => {
    registerCustomKerasLayer('test', () => []);
    expect(getCustomKerasLayerEmitter('test')).toBeDefined();
  });
});
