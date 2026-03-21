// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Toolbar } from '../src/components/toolbar.js';

describe('Toolbar', () => {
  let container: HTMLElement;

  beforeEach(() => {
    container = document.createElement('div');
  });

  it('renders buttons and dispatches events', () => {
    const onCleanGraph = vi.fn();
    const onMakeDynamic = vi.fn();
    const onStripInitializers = vi.fn();

    new Toolbar(container, {
      onCleanGraph,
      onMakeDynamic,
      onStripInitializers,
      onFixMixedPrecision: vi.fn(),
      onRemoveTrainingNodes: vi.fn(),
      onFoldConstants: vi.fn(),
      onExtractWeights: vi.fn(),
      onSanitizeNames: vi.fn(),
      onValidateOpset: vi.fn(),
      onValidateGraph: vi.fn(),
      onAutoFix: vi.fn(),
      onDeduplicateConstants: vi.fn(),
      onExportStats: vi.fn(),
      onExportGraphJSON: vi.fn(),
      onFeedback: vi.fn(),
      onToggleStrict: vi.fn(),
      onSaveSession: vi.fn(),
      onExportModel: vi.fn(),
    });

    const buttons = container.querySelectorAll('button');
    expect(buttons.length).toBe(17);
    expect(container.querySelectorAll('input').length).toBe(1);

    buttons[0]!.click();
    expect(onCleanGraph).toHaveBeenCalledTimes(1);

    buttons[1]!.click();
    expect(onMakeDynamic).toHaveBeenCalledTimes(1);

    buttons[2]!.click();
    expect(onStripInitializers).toHaveBeenCalledTimes(1);
  });
});
