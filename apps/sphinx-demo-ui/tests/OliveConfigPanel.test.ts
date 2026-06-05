// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { OliveConfigPanel } from '../src/components/OliveConfigPanel.js';

describe('OliveConfigPanel', () => {
  it('should render and handle changes', () => {
    const panel = new OliveConfigPanel();
    document.body.appendChild(panel.element);

    const checkbox = panel.element.querySelector('.demo-olive-fusion-checkbox') as HTMLInputElement;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event('change'));

    expect(panel.getConfig().enableTransformerFusion).toBe(true);
  });
});
