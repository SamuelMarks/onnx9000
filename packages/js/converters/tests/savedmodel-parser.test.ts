import { describe, it, expect } from 'vitest';
import { parseSavedModel } from '../src/keras/savedmodel-parser.js';

describe('SavedModel Parser', () => {
  it('throws on missing saved_model.pb', () => {
    expect(() => parseSavedModel({ 'variables.index': new Uint8Array() })).toThrow(
      'Invalid SavedModel format: missing saved_model.pb',
    );
  });

  it('parses correctly', () => {
    const files = {
      'my_model/saved_model.pb': new Uint8Array([1, 2, 3]),
      'my_model/variables/variables.index': new Uint8Array([4, 5]),
      'my_model/variables/variables.data-00000-of-00001': new Uint8Array([6, 7]),
      'my_model/variables/variables.data-00001-of-00001': new Uint8Array([8, 9]),
    };
    const res = parseSavedModel(files);
    expect(res.savedModelPb).toEqual(new Uint8Array([1, 2, 3]));
    expect(res.variablesIndex).toEqual(new Uint8Array([4, 5]));
    expect(res.variablesData).toHaveLength(2);
  });
});
