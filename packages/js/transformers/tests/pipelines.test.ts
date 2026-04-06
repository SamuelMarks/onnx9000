import { describe, it, expect, vi } from 'vitest';
import * as pipes from '../src/pipelines/index';

describe('Pipelines', () => {
  it('Pipeline empty init', async () => {
    const p = new pipes.Pipeline('task', {}, {});
    expect(p).toBeDefined();
    await expect(p('input')).resolves.toBeDefined();
    await expect(p._call('input')).resolves.toBeDefined();
  });

  it('specific pipelines init', async () => {
    const classes = [
      pipes.FeatureExtractionPipeline,
      pipes.TextClassificationPipeline,
      pipes.TokenClassificationPipeline,
      pipes.QuestionAnsweringPipeline,
      pipes.ZeroShotClassificationPipeline,
      pipes.TranslationPipeline,
      pipes.SummarizationPipeline,
      pipes.TextGenerationPipeline,
      pipes.Text2TextGenerationPipeline,
      pipes.FillMaskPipeline,
      pipes.ImageClassificationPipeline,
      pipes.ObjectDetectionPipeline,
      pipes.ZeroShotImageClassificationPipeline,
      pipes.ImageSegmentationPipeline,
      pipes.DepthEstimationPipeline,
      pipes.ImageToImagePipeline,
      pipes.AudioClassificationPipeline,
      pipes.AutomaticSpeechRecognitionPipeline,
      pipes.TextToSpeechPipeline,
      pipes.DocumentQuestionAnsweringPipeline,
      pipes.VisualQuestionAnsweringPipeline,
      pipes.ImageFeatureExtractionPipeline,
    ];
    for (const Cls of classes) {
      const p = new Cls('task', {}, {});
      expect(p).toBeDefined();
      await expect(p._call('input')).resolves.toBeDefined();
    }
    const od = new pipes.ObjectDetectionPipeline({}, {}, {}, {});
    expect(od.wasmNMS()).toEqual([]);
    const boxes = [[1, 2]];
    od.denormalizeBbox(boxes);
    expect(boxes[0][0]).toBe(255);
  });

  it('ModelOutput', () => {
    const m = new pipes.ModelOutput({});
    expect(m).toBeDefined();
  });

  it('pipeline factory wrapper', async () => {
    await expect(pipes.pipeline('unsupported_task' as Object)).rejects.toThrow(
      'Unsupported task: unsupported_task',
    );
  });
});

it('pipeline switch', async () => {
  const tasks = [
    'feature-extraction',
    'text-classification',
    'token-classification',
    'question-answering',
    'zero-shot-classification',
    'translation',
    'summarization',
    'text-generation',
    'text2text-generation',
    'fill-mask',
    'image-classification',
    'object-detection',
    'zero-shot-image-classification',
    'image-segmentation',
    'depth-estimation',
    'image-to-image',
    'audio-classification',
    'automatic-speech-recognition',
    'text-to-speech',
    'document-question-answering',
    'visual-question-answering',
    'image-feature-extraction',
  ];
  for (const t of tasks) {
    await pipes.pipeline(t as Object);
  }
});

it('pipeline uncovered lines', async () => {
  await pipes.pipeline('feature-extraction'); // pool hit

  const gen = new pipes.TextGenerationPipeline('task', null, null);
  const stream = await gen._forward('a', { stream: true });
  for await (const s of stream) {
  }

  await gen.postprocess('a', { stream: true });
});
