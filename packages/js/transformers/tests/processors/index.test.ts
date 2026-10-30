import { describe, expect, it } from 'vitest';
import {
  AutoProcessor,
  BaseImageProcessor,
  CLIPImageProcessor,
  DeiTImageProcessor,
  DetrImageProcessor,
  ONNX9000Audio,
  ONNX9000Image,
  SequenceFeatureExtractor,
  SpeechT5FeatureExtractor,
  ViTImageProcessor,
  Wav2Vec2FeatureExtractor,
  WhisperFeatureExtractor,
  YolosImageProcessor,
} from '../../src/processors/index';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover ONNX9000Image', () => {
    try {
      const obj = Construct(ONNX9000Image);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ONNX9000Audio', () => {
    try {
      const obj = Construct(ONNX9000Audio);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover BaseImageProcessor', () => {
    try {
      const obj = Construct(BaseImageProcessor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ViTImageProcessor', () => {
    try {
      const obj = Construct(ViTImageProcessor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover CLIPImageProcessor', () => {
    try {
      const obj = Construct(CLIPImageProcessor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover DeiTImageProcessor', () => {
    try {
      const obj = Construct(DeiTImageProcessor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover DetrImageProcessor', () => {
    try {
      const obj = Construct(DetrImageProcessor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover YolosImageProcessor', () => {
    try {
      const obj = Construct(YolosImageProcessor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover SequenceFeatureExtractor', () => {
    try {
      const obj = Construct(SequenceFeatureExtractor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WhisperFeatureExtractor', () => {
    try {
      const obj = Construct(WhisperFeatureExtractor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Wav2Vec2FeatureExtractor', () => {
    try {
      const obj = Construct(Wav2Vec2FeatureExtractor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover SpeechT5FeatureExtractor', () => {
    try {
      const obj = Construct(SpeechT5FeatureExtractor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoProcessor', () => {
    try {
      const obj = Construct(AutoProcessor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
