// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { PromoteButton } from '../src/components/PromoteButton.js';
import { globalEventBus } from '../src/core/EventBus.js';

describe('PromoteButton', () => {
  it('should render and react to events', () => {
    const btn = new PromoteButton();
    btn.mount(document.body);

    expect(btn.element.disabled).toBe(true);

    globalEventBus.emit('TARGET_ARTIFACT_GENERATED', {});
    expect(btn.element.disabled).toBe(false);
  });
});
