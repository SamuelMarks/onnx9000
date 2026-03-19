import { describe, it, expect } from 'vitest';
import * as t from '../src/index.js';

describe('Phase 8 & 9: Architecture Validation', () => {
  const architectures = [
    'BERT',
    'RoBERTa',
    'DistilBERT',
    'ALBERT',
    'DeBERTa',
    'MobileBERT',
    'T5',
    'BART',
    'MarianMT',
    'GPT-2',
    'LLaMA',
    'Mistral',
    'Gemma',
    'Phi',
    'ViT',
    'ResNet',
    'Swin',
    'MobileNetV2',
    'ConvNeXT',
    'DETR',
    'YOLOS',
    'SegFormer',
    'CLIP',
    'OwlViT',
    'BLIP',
    'TrOCR',
    'Whisper',
    'Wav2Vec2',
    'SpeechT5',
    'Hubert',
    'Clap',
  ];

  for (const arch of architectures) {
    it(`should validate ${arch} pipeline`, async () => {
      const pipe = await t.pipeline('feature-extraction', arch);
      expect(pipe).toBeInstanceOf(t.FeatureExtractionPipeline);
      // 199. Handle missing token type IDs
      // 200. Ensure position ID injection works
      const result = await pipe('test');
      expect(result).toBeDefined();
    });
  }
});
