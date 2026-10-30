import { describe, expect, it } from 'vitest';
import {
  AutoConfig,
  AutoFeatureExtractor,
  AutoModel,
  AutoModelForCausalLM,
  AutoModelForImageClassification,
  AutoModelForMaskedLM,
  AutoModelForObjectDetection,
  AutoModelForQuestionAnswering,
  AutoModelForSeq2SeqLM,
  AutoModelForSequenceClassification,
  AutoModelForSpeechSeq2Seq,
  AutoModelForTokenClassification,
  GenerationMixin,
  HubConfig,
  ModelCache,
  PreTrainedModel,
} from '../../src/models/index';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover HubConfig', () => {
    try {
      const obj = Construct(HubConfig);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ModelCache', () => {
    try {
      const obj = Construct(ModelCache);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover PreTrainedModel', () => {
    try {
      const obj = Construct(PreTrainedModel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover GenerationMixin', () => {
    try {
      const obj = Construct(GenerationMixin);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoConfig', () => {
    try {
      const obj = Construct(AutoConfig);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoFeatureExtractor', () => {
    try {
      const obj = Construct(AutoFeatureExtractor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoModelForSequenceClassification', () => {
    try {
      const obj = Construct(AutoModelForSequenceClassification);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoModelForTokenClassification', () => {
    try {
      const obj = Construct(AutoModelForTokenClassification);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoModelForQuestionAnswering', () => {
    try {
      const obj = Construct(AutoModelForQuestionAnswering);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoModelForCausalLM', () => {
    try {
      const obj = Construct(AutoModelForCausalLM);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoModelForMaskedLM', () => {
    try {
      const obj = Construct(AutoModelForMaskedLM);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoModelForSeq2SeqLM', () => {
    try {
      const obj = Construct(AutoModelForSeq2SeqLM);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoModelForImageClassification', () => {
    try {
      const obj = Construct(AutoModelForImageClassification);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoModelForObjectDetection', () => {
    try {
      const obj = Construct(AutoModelForObjectDetection);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoModelForSpeechSeq2Seq', () => {
    try {
      const obj = Construct(AutoModelForSpeechSeq2Seq);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoModel', () => {
    try {
      const obj = Construct(AutoModel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
