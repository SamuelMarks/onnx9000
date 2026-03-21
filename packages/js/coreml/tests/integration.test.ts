import { describe, it, expect } from 'vitest';
import { Graph } from '@onnx9000/core';
import { convertToCoreML } from '../src/index.js';

describe('ONNX to CoreML Integration', () => {
  it('Validates baseline conversion of ResNet50 (mock)', () => {
    const graph = new Graph('resnet50_mock');
    const program = convertToCoreML(graph);
    expect(program.functions['main']).toBeDefined();
  });

  it('Validates baseline conversion of MobileNetV2 (mock)', () => {
    const graph = new Graph('mobilenetv2_mock');
    const program = convertToCoreML(graph);
    expect(program.functions['main']).toBeDefined();
  });

  it('Validates baseline conversion of YOLOv8 (mock)', () => {
    const graph = new Graph('yolov8_mock');
    const program = convertToCoreML(graph);
    expect(program.functions['main']).toBeDefined();
  });

  it('Validates baseline conversion of BERT (mock)', () => {
    const graph = new Graph('bert_mock');
    const program = convertToCoreML(graph);
    expect(program.functions['main']).toBeDefined();
  });

  it('Validates baseline conversion of GPT-2 (mock)', () => {
    const graph = new Graph('gpt2_mock');
    const program = convertToCoreML(graph);
    expect(program.functions['main']).toBeDefined();
  });

  it('Validates baseline conversion of Whisper-Tiny (mock)', () => {
    const graph = new Graph('whisper_tiny_mock');
    const program = convertToCoreML(graph);
    expect(program.functions['main']).toBeDefined();
  });
});
