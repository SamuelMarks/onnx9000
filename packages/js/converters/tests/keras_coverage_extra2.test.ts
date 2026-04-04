import { describe, it, expect } from 'vitest';
import { Keras2OnnxConverter } from '../src/keras/index';

describe('Coverage Keras Converter 2', () => {
  it('handleEinsumDense', async () => {
    const kerasJson = {
      format: 'layers-model',
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            {
              name: 'einsum',
              class_name: 'EinsumDense',
              config: { equation: 'abc,cd->abd', bias_axes: 'd' },
            },
            {
              name: 'einsum2',
              class_name: 'EinsumDense',
              config: { equation: 'abc,cd->abd', bias_axes: null },
            },
          ],
        },
      },
      weightsManifest: [],
    };

    const converter = new Keras2OnnxConverter(JSON.stringify(kerasJson), new Map());
    // Since convert is heavily tested, we can just call handleEinsumDense directly for coverage
    const nodes1 = (converter as any).handleEinsumDense(
      'in',
      'out',
      'einsum',
      'einsum_1',
      kerasJson.modelTopology.config.layers[0].config,
    );
    expect(nodes1.find((n: any) => n.opType === 'Add')).toBeDefined();

    const nodes2 = (converter as any).handleEinsumDense(
      'in',
      'out',
      'einsum2',
      'einsum_2',
      kerasJson.modelTopology.config.layers[1].config,
    );
    expect(nodes2.find((n: any) => n.opType === 'Identity')).toBeDefined();
  });
});
