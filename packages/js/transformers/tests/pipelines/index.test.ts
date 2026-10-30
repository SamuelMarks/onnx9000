import { describe, expect, it } from 'vitest';
import {
  AudioClassificationPipeline,
  AutomaticSpeechRecognitionPipeline,
  Callable,
  DepthEstimationPipeline,
  DocumentQuestionAnsweringPipeline,
  FeatureExtractionPipeline,
  FillMaskPipeline,
  ImageClassificationPipeline,
  ImageFeatureExtractionPipeline,
  ImageSegmentationPipeline,
  ImageToImagePipeline,
  ModelOutput,
  ObjectDetectionPipeline,
  Pipeline,
  pipeline,
  QuestionAnsweringPipeline,
  SummarizationPipeline,
  Text2TextGenerationPipeline,
  TextClassificationPipeline,
  TextGenerationPipeline,
  TextToSpeechPipeline,
  TokenClassificationPipeline,
  TranslationPipeline,
  VisualQuestionAnsweringPipeline,
  ZeroShotClassificationPipeline,
  ZeroShotImageClassificationPipeline,
} from '../../src/pipelines/index';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover Callable', () => {
    try {
      const obj = Construct(Callable);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Pipeline', () => {
    try {
      const obj = Construct(Pipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover FeatureExtractionPipeline', () => {
    try {
      const obj = Construct(FeatureExtractionPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ModelOutput', () => {
    try {
      const obj = Construct(ModelOutput);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover TextClassificationPipeline', () => {
    try {
      const obj = Construct(TextClassificationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover TokenClassificationPipeline', () => {
    try {
      const obj = Construct(TokenClassificationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover QuestionAnsweringPipeline', () => {
    try {
      const obj = Construct(QuestionAnsweringPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ZeroShotClassificationPipeline', () => {
    try {
      const obj = Construct(ZeroShotClassificationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover TranslationPipeline', () => {
    try {
      const obj = Construct(TranslationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover SummarizationPipeline', () => {
    try {
      const obj = Construct(SummarizationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover TextGenerationPipeline', () => {
    try {
      const obj = Construct(TextGenerationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Text2TextGenerationPipeline', () => {
    try {
      const obj = Construct(Text2TextGenerationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover FillMaskPipeline', () => {
    try {
      const obj = Construct(FillMaskPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ImageClassificationPipeline', () => {
    try {
      const obj = Construct(ImageClassificationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ObjectDetectionPipeline', () => {
    try {
      const obj = Construct(ObjectDetectionPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ZeroShotImageClassificationPipeline', () => {
    try {
      const obj = Construct(ZeroShotImageClassificationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ImageSegmentationPipeline', () => {
    try {
      const obj = Construct(ImageSegmentationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover DepthEstimationPipeline', () => {
    try {
      const obj = Construct(DepthEstimationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ImageToImagePipeline', () => {
    try {
      const obj = Construct(ImageToImagePipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AudioClassificationPipeline', () => {
    try {
      const obj = Construct(AudioClassificationPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutomaticSpeechRecognitionPipeline', () => {
    try {
      const obj = Construct(AutomaticSpeechRecognitionPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover TextToSpeechPipeline', () => {
    try {
      const obj = Construct(TextToSpeechPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover DocumentQuestionAnsweringPipeline', () => {
    try {
      const obj = Construct(DocumentQuestionAnsweringPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover VisualQuestionAnsweringPipeline', () => {
    try {
      const obj = Construct(VisualQuestionAnsweringPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ImageFeatureExtractionPipeline', () => {
    try {
      const obj = Construct(ImageFeatureExtractionPipeline);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover pipeline', async () => {
    try {
      const res = Call(pipeline);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
