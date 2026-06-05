// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { PromoteButton } from '../src/components/PromoteButton.js';
import { globalEventBus } from '../src/core/EventBus.js';

describe('PromoteButton', () => {
  it('should render and react to events', () => {
    const btn = new PromoteButton();
    document.body.appendChild(btn.element);

    expect(btn.element.disabled).toBe(true);

    globalEventBus.emit('TARGET_ARTIFACT_GENERATED', {});
    expect(btn.element.disabled).toBe(false);
  });
});
