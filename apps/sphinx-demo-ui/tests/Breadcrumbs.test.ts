// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { Breadcrumbs } from '../src/components/Breadcrumbs.js';
import { globalEventBus } from '../src/core/EventBus.js';
import type { PipelineNode } from '../src/core/PipelineNode.js';

describe('Breadcrumbs', () => {
  it('should render and react to events', () => {
    const breadcrumbs = new Breadcrumbs();
    breadcrumbs.mount(document.body);

    const node: PipelineNode = { id: '1', description: 'test node' } as any;
    globalEventBus.emit('PIPELINE_STEP_ADDED', node);

    expect(breadcrumbs.element.innerHTML).toContain('test node');
  });
});
