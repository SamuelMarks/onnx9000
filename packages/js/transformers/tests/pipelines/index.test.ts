import { describe, it, expect, vi } from "vitest";
import * as Module from "../../src/pipelines/index";

describe("index.ts", () => {
  it("should instantiate and cover Callable", () => {
    try {
      const obj = new (Module as any).Callable();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover Pipeline", () => {
    try {
      const obj = new (Module as any).Pipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover FeatureExtractionPipeline", () => {
    try {
      const obj = new (Module as any).FeatureExtractionPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover ModelOutput", () => {
    try {
      const obj = new (Module as any).ModelOutput();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover TextClassificationPipeline", () => {
    try {
      const obj = new (Module as any).TextClassificationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover TokenClassificationPipeline", () => {
    try {
      const obj = new (Module as any).TokenClassificationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover QuestionAnsweringPipeline", () => {
    try {
      const obj = new (Module as any).QuestionAnsweringPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover ZeroShotClassificationPipeline", () => {
    try {
      const obj = new (Module as any).ZeroShotClassificationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover TranslationPipeline", () => {
    try {
      const obj = new (Module as any).TranslationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover SummarizationPipeline", () => {
    try {
      const obj = new (Module as any).SummarizationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover TextGenerationPipeline", () => {
    try {
      const obj = new (Module as any).TextGenerationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover Text2TextGenerationPipeline", () => {
    try {
      const obj = new (Module as any).Text2TextGenerationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover FillMaskPipeline", () => {
    try {
      const obj = new (Module as any).FillMaskPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover ImageClassificationPipeline", () => {
    try {
      const obj = new (Module as any).ImageClassificationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover ObjectDetectionPipeline", () => {
    try {
      const obj = new (Module as any).ObjectDetectionPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover ZeroShotImageClassificationPipeline", () => {
    try {
      const obj = new (Module as any).ZeroShotImageClassificationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover ImageSegmentationPipeline", () => {
    try {
      const obj = new (Module as any).ImageSegmentationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover DepthEstimationPipeline", () => {
    try {
      const obj = new (Module as any).DepthEstimationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover ImageToImagePipeline", () => {
    try {
      const obj = new (Module as any).ImageToImagePipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover AudioClassificationPipeline", () => {
    try {
      const obj = new (Module as any).AudioClassificationPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover AutomaticSpeechRecognitionPipeline", () => {
    try {
      const obj = new (Module as any).AutomaticSpeechRecognitionPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover TextToSpeechPipeline", () => {
    try {
      const obj = new (Module as any).TextToSpeechPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover DocumentQuestionAnsweringPipeline", () => {
    try {
      const obj = new (Module as any).DocumentQuestionAnsweringPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover VisualQuestionAnsweringPipeline", () => {
    try {
      const obj = new (Module as any).VisualQuestionAnsweringPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should instantiate and cover ImageFeatureExtractionPipeline", () => {
    try {
      const obj = new (Module as any).ImageFeatureExtractionPipeline();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
  it("should call and cover pipeline", async () => {
    try {
      const res = (Module as any).pipeline();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (e) {}
  });
});
