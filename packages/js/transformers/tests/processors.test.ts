import { describe, expect, it } from 'vitest';
import {
  AutoProcessor,
  BaseImageProcessor,
  SequenceFeatureExtractor,
} from '../src/processors/index.js';

describe('processors', () => {
  it('should process image', async () => {
    const proc = new BaseImageProcessor();
    const res = await proc.process('image');
    expect(res.pixel_values).toBeDefined();
  });

  it('should process audio', async () => {
    const proc = new SequenceFeatureExtractor();
    const res = await proc.process('audio');
    expect(res.input_features).toBeDefined();
  });

  it('should auto process', async () => {
    const proc = await AutoProcessor.fromPretrained('test');
    const res = await proc.process('image');
    expect(res).toBeDefined();
  });
});
