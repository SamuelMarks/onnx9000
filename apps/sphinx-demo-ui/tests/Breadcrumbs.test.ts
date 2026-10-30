import { describe, expect, it } from 'vitest';
import { Breadcrumbs } from '../src/components/Breadcrumbs.js';
import { globalEventBus } from '../src/core/EventBus.js';
import { PipelineNode } from '../src/core/PipelineNode.js';

describe('Breadcrumbs', () => {
  it('should render and react to events', () => {
    const breadcrumbs = new Breadcrumbs();
    breadcrumbs.mount(document.body);

    const node = new PipelineNode(
      { sourceFramework: 'onnx', targetFramework: 'tflite', activeFile: 'model.onnx' },
      'test node',
    );
    globalEventBus.emit('PIPELINE_STEP_ADDED', node);

    expect(breadcrumbs.element.innerHTML).toContain('test node');
  });
});
