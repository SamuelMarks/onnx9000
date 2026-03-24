import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Breadcrumbs } from '../../src/components/Breadcrumbs';
import { PipelineNode } from '../../src/core/PipelineNode';
import { globalEventBus } from '../../src/core/EventBus';

describe('Breadcrumbs', () => {
  beforeEach(() => {
    globalEventBus.clearAll();
  });

  it('should render a placeholder when empty', () => {
    const breadcrumbs = new Breadcrumbs();
    const el = (breadcrumbs as any).element as HTMLElement;

    expect(el.className).toBe('demo-breadcrumbs-container');
    const placeholder = el.querySelector('.demo-breadcrumb-placeholder');
    expect(placeholder).not.toBeNull();
    expect(placeholder?.textContent).toContain('Pipeline is empty');
  });

  it('should add items on PIPELINE_STEP_ADDED event', () => {
    const breadcrumbs = new Breadcrumbs();
    const el = (breadcrumbs as any).element as HTMLElement;
    breadcrumbs.mount(document.body);

    const node1 = new PipelineNode(
      { sourceFramework: 'k', targetFramework: 't', activeFile: '' },
      'Step 1'
    );
    const node2 = new PipelineNode(
      { sourceFramework: 't', targetFramework: 'm', activeFile: '' },
      'Step 2'
    );

    globalEventBus.emit('PIPELINE_STEP_ADDED', node1);
    globalEventBus.emit('PIPELINE_STEP_ADDED', node2);

    const items = el.querySelectorAll('.demo-breadcrumb-item');
    expect(items.length).toBe(2);
    expect(items[0].textContent).toBe('Step 1');
    expect(items[1].textContent).toBe('Step 2');
    expect(items[1].classList.contains('active')).toBe(true);

    const separators = el.querySelectorAll('.demo-breadcrumb-separator');
    expect(separators.length).toBe(1);

    breadcrumbs.unmount();
  });

  it('should remove items on PIPELINE_STEP_REMOVED event', () => {
    const breadcrumbs = new Breadcrumbs();
    const el = (breadcrumbs as any).element as HTMLElement;
    breadcrumbs.mount(document.body);

    const node1 = new PipelineNode(
      { sourceFramework: 'k', targetFramework: 't', activeFile: '' },
      'Step 1'
    );
    const node2 = new PipelineNode(
      { sourceFramework: 't', targetFramework: 'm', activeFile: '' },
      'Step 2'
    );
    const node3 = new PipelineNode(
      { sourceFramework: 'm', targetFramework: 'i', activeFile: '' },
      'Step 3'
    );

    globalEventBus.emit('PIPELINE_STEP_ADDED', node1);
    globalEventBus.emit('PIPELINE_STEP_ADDED', node2);
    globalEventBus.emit('PIPELINE_STEP_ADDED', node3);

    expect(el.querySelectorAll('.demo-breadcrumb-item').length).toBe(3);

    // Simulate undoing Step 3
    globalEventBus.emit('PIPELINE_STEP_REMOVED', node3);

    const remainingItems = el.querySelectorAll('.demo-breadcrumb-item');
    expect(remainingItems.length).toBe(2);
    expect(remainingItems[1].textContent).toBe('Step 2');

    breadcrumbs.unmount();
  });

  it('should safely ignore removing a non-existent node', () => {
    const breadcrumbs = new Breadcrumbs();
    const el = (breadcrumbs as any).element as HTMLElement;
    breadcrumbs.mount(document.body);

    const node1 = new PipelineNode(
      { sourceFramework: 'k', targetFramework: 't', activeFile: '' },
      'Step 1'
    );
    globalEventBus.emit('PIPELINE_STEP_ADDED', node1);

    const dummyNode = new PipelineNode(
      { sourceFramework: 'm', targetFramework: 'i', activeFile: '' },
      'Missing'
    );
    globalEventBus.emit('PIPELINE_STEP_REMOVED', dummyNode);

    // Should still have 1 item
    expect(el.querySelectorAll('.demo-breadcrumb-item').length).toBe(1);

    breadcrumbs.unmount();
  });

  it('should request revert on item click', () => {
    const breadcrumbs = new Breadcrumbs();
    breadcrumbs.mount(document.body);
    const el = (breadcrumbs as any).element as HTMLElement;

    const node1 = new PipelineNode(
      { sourceFramework: 'k', targetFramework: 't', activeFile: '' },
      'Step 1'
    );
    const node2 = new PipelineNode(
      { sourceFramework: 't', targetFramework: 'm', activeFile: '' },
      'Step 2'
    );

    globalEventBus.emit('PIPELINE_STEP_ADDED', node1);
    globalEventBus.emit('PIPELINE_STEP_ADDED', node2);

    const revertSpy = vi.fn();
    globalEventBus.on('PIPELINE_REVERT_REQUESTED', revertSpy);

    const firstBtn = el.querySelectorAll('.demo-breadcrumb-item')[0] as HTMLButtonElement;
    firstBtn.click();

    expect(revertSpy).toHaveBeenCalledWith(node1.id);

    breadcrumbs.unmount();
  });
});
