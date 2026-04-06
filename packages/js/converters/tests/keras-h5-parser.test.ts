import { describe, it, expect, vi } from 'vitest';
import { parseKerasH5 } from '../src/keras/h5-parser.js';

vi.mock('jsfive', () => {
  return {
    File: class MockFile {
      attrs: Object;
      _groups: Object;
      keys: string[];
      constructor(buf: Object, name: string) {
        // Determine mock behavior based on buffer length
        if (buf.byteLength === 0) {
          this.attrs = {}; // No model_config
        } else if (buf.byteLength === 1) {
          this.attrs = {
            model_config: '{"mock": "config"}',
            keras_version: '2.0',
            backend: 'tf',
          };
          this._groups = {
            model_weights: {
              keys: ['dense_1'],
              attrs: {},
              get: (k: string) => {
                if (k === 'dense_1') {
                  return {
                    keys: ['kernel:0'],
                    attrs: { weight_names: ['kernel:0'] },
                    get: (k2: string) => {
                      return { shape: [2, 2], value: new Float32Array([1, 2, 3, 4]) };
                    },
                  };
                }
              },
            },
          };
        } else if (buf.byteLength === 2) {
          this.attrs = {
            model_config: new TextEncoder().encode('{"mock": "config2"}'),
            keras_version: new TextEncoder().encode('2.1'),
            backend: new TextEncoder().encode('theano'),
          };
          // No model weights group
          this.keys = ['layer1'];
          this._groups = {};
        } else if (buf.byteLength === 3) {
          this.attrs = { model_config: '{}' };
          this._groups = {
            model_weights: {
              keys: ['nested_layer'],
              attrs: {},
              get: (k: string) => {
                return {
                  keys: ['nested_weight'],
                  attrs: { weight_names: 'nested_weight' },
                  get: (k2: string) => {
                    return {
                      keys: ['actual_dataset'], // inner group
                      get: (k3: string) => ({ shape: [1], value: new Float32Array([1]) }),
                    };
                  },
                };
              },
            },
          };
        } else if (buf.byteLength === 4) {
          this.attrs = { model_config: '{}' };
          this._groups = {
            model_weights: {
              keys: ['layer_error'],
              get: () => {
                throw new Error('get error');
              },
            },
          };
        }
      }

      get(k: string) {
        if (this._groups[k]) return this._groups[k];
        throw new Error('Not found');
      }
    },
  };
});

describe('Keras H5 Parser', () => {
  it('throws if no model_config', () => {
    expect(() => parseKerasH5(new ArrayBuffer(0))).toThrow(
      'HDF5 file does not contain a Keras model_config attribute',
    );
  });

  it('parses valid model with string attrs and weights', () => {
    const model = parseKerasH5(new ArrayBuffer(1));
    expect(model.modelConfig).toEqual({ mock: 'config' });
    expect(model.kerasVersion).toBe('2.0');
    expect(model.backend).toBe('tf');
    expect(model.weights['kernel:0'].shape).toEqual([2, 2]);
  });

  it('parses valid model with Uint8Array attrs and no model_weights group', () => {
    const model = parseKerasH5(new ArrayBuffer(2));
    expect(model.modelConfig).toEqual({ mock: 'config2' });
    expect(model.kerasVersion).toBe('2.1');
    expect(model.backend).toBe('theano');
  });

  it('handles nested layer groups and string weight_names', () => {
    const model = parseKerasH5(new ArrayBuffer(3));
    expect(model.weights['nested_weight'].shape).toEqual([1]);
  });

  it('handles errors getting layer groups', () => {
    const model = parseKerasH5(new ArrayBuffer(4));
    expect(Object.keys(model.weights).length).toBe(0);
  });
});
