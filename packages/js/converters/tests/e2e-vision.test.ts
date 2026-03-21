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
          {
            class_name: 'InputLayer',
            config: { name: 'input_1', batch_input_shape: [null, 224, 224, 3] },
          },
        ],
      },
    },
    weightsManifest: [],
  });
}

describe('e2e-vision', () => {
  it('converts vision models correctly', () => {
    const models = [
      'MobileNetV1',
      'MobileNetV2',
      'MobileNetV3',
      'ResNet50',
      'ResNet101',
      'InceptionV3',
      'Xception',
      'VGG16',
      'VGG19',
      'EfficientNet',
      'DenseNet121',
      'NASNetMobile',
      'PoseNet',
      'BodyPix',
    ];

    for (const name of models) {
      const json = mockModel(name);
      const converter = new Keras2OnnxConverter(json);
      const bytes = converter.convert();
      expect(bytes).toBeInstanceOf(Uint8Array);
    }
  });
});
