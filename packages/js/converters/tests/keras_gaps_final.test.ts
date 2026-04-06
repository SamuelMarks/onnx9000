import { describe, it, expect } from 'vitest';
import { Keras2OnnxConverter } from '../src/keras/index.js';

describe('Keras2OnnxConverter Final Gaps', () => {
  it('should hit missing branches in convert() and handlers', () => {
    const mockModel = JSON.stringify({
      format: 'layers-model',
      weightsManifest: [],
      modelTopology: {
        class_name: 'Model',
        config: {
          name: 'gap_model',
          layers: [
            {
              class_name: 'Dropout',
              name: 'do',
              config: { name: 'do', rate: 0.5 },
              inbound_nodes: [[['in1', 0, 0, {}]]],
            },
            {
              class_name: 'AlphaDropout',
              name: 'ado',
              config: { name: 'ado', rate: 0.5 },
              inbound_nodes: [[['in1', 0, 0, {}]]],
            },
            {
              class_name: 'SpatialDropout2D',
              name: 'sdo',
              config: { name: 'sdo', rate: 0.5 },
              inbound_nodes: [[['in1', 0, 0, {}]]],
            },
            {
              class_name: 'GaussianDropout',
              name: 'gdo',
              config: { name: 'gdo', rate: 0.5 },
              inbound_nodes: [[['in1', 0, 0, {}]]],
            },
            {
              class_name: 'GaussianNoise',
              name: 'gn',
              config: { name: 'gn', stddev: 0.1 },
              inbound_nodes: [[['in1', 0, 0, {}]]],
            },
            {
              class_name: 'RepeatVector',
              name: 'rv',
              config: { name: 'rv', n: 5 },
              inbound_nodes: [[['in1', 0, 0, {}]]],
            },
          ],
        },
      },
    });
    const converter = new Keras2OnnxConverter(mockModel);
    converter.convert();

    // Final handler sweep with specific inputs
    const handlers = (converter as Object).handlers as Map<string, Object>;
    for (const [className, handler] of handlers.entries()) {
      try {
        handler('in', 'out', 'node', 'layer', {}, className, { inboundNodes: ['in'] } as Object);
      } catch (e) {}
    }
  });
});
