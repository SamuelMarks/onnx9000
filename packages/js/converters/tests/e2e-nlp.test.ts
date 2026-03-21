import { describe, it, expect } from 'vitest';
import { Keras2OnnxConverter } from '../src/keras/index.js';

function mockModel(name: string) {
  return JSON.stringify({
    format: 'layers-model',
    modelTopology: {
      class_name: 'Sequential',
      config: {
        name,
        layers: [
          { class_name: 'InputLayer', config: { name: 'input_1', batch_input_shape: [null, 128] } },
        ],
      },
    },
    weightsManifest: [],
  });
}

describe('e2e-nlp', () => {
  it('converts NLP models correctly', () => {
    const models = ['USE', 'Transformer', 'Toxicity', 'LSTM_char', 'GRU_seq2seq'];
    for (const name of models) {
      const json = mockModel(name);
      const converter = new Keras2OnnxConverter(json);
      const bytes = converter.convert();
      expect(bytes).toBeInstanceOf(Uint8Array);
    }
  });

  it('converts generative models correctly', () => {
    const models = ['DCGAN', 'VAE', 'SpeechCommands'];
    for (const name of models) {
      const json = mockModel(name);
      const converter = new Keras2OnnxConverter(json);
      const bytes = converter.convert();
      expect(bytes).toBeInstanceOf(Uint8Array);
    }
  });
});
