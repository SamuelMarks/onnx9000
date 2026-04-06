import { describe, it, expect } from 'vitest';
import * as models from '../src/models/index';

describe('Models', () => {
  it('HubConfig', () => {
    models.HubConfig.setEndpoint('test');
    expect(models.HubConfig.endpoint).toBe('test');
    models.HubConfig.setApiKey('key');
    expect(models.HubConfig.apiKey).toBe('key');
  });

  it('Model classes', async () => {
    const list = [
      models.PreTrainedModel,
      models.GenerationMixin,
      models.AutoModelForSequenceClassification,
      models.AutoModelForTokenClassification,
      models.AutoModelForQuestionAnswering,
      models.AutoModelForCausalLM,
      models.AutoModelForMaskedLM,
      models.AutoModelForSeq2SeqLM,
      models.AutoModelForImageClassification,
      models.AutoModelForObjectDetection,
      models.AutoModelForSpeechSeq2Seq,
    ];
    for (const exp of list) {
      const inst = new (exp as Object)({ myconfig: 1 }, 'model');
      expect(inst.config).toBeDefined();
      if (inst.generate) {
        await expect(inst.generate()).resolves.toBeDefined();
      }
      if (inst.forward) {
        await expect(inst.forward()).resolves.toBeDefined();
      }
    }
  });

  it('AutoModels', async () => {
    await expect(models.AutoModel.fromPretrained('a')).resolves.toBeDefined();
    await expect(models.AutoModelForCausalLM.fromPretrained('a')).resolves.toBeDefined();
    await expect(
      models.AutoModelForSequenceClassification.fromPretrained('a'),
    ).resolves.toBeDefined();
    await expect(models.AutoModelForTokenClassification.fromPretrained('a')).resolves.toBeDefined();
    await expect(models.AutoModelForQuestionAnswering.fromPretrained('a')).resolves.toBeDefined();
    await expect(models.AutoModelForMaskedLM.fromPretrained('a')).resolves.toBeDefined();
    await expect(models.AutoModelForSeq2SeqLM.fromPretrained('a')).resolves.toBeDefined();
    await expect(models.AutoModelForImageClassification.fromPretrained('a')).resolves.toBeDefined();
    await expect(models.AutoModelForObjectDetection.fromPretrained('a')).resolves.toBeDefined();
    await expect(models.AutoModelForSpeechSeq2Seq.fromPretrained('a')).resolves.toBeDefined();
  });

  it('ModelCache and config/feature', async () => {
    expect(models.ModelCache).toBeDefined();
    await expect(models.AutoConfig.fromPretrained('id')).resolves.toBeDefined();
    await expect(models.AutoFeatureExtractor.fromPretrained('id')).resolves.toBeDefined();
  });
});

it('uncovered model methods', async () => {
  await models.ModelCache.clearCache();
  await models.ModelCache.getFromCache('a');
  await models.ModelCache.putInCache('a', 'b');

  const inst = new models.PreTrainedModel({}, 'path');
  await inst.init();
  inst.dispose();
});
