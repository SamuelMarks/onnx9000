import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OliveConfigPanel } from '../../src/components/OliveConfigPanel';
import { globalEventBus } from '../../src/core/EventBus';

describe('OliveConfigPanel', () => {
  beforeEach(() => {
    globalEventBus.clearAll();
  });

  it('should render properly with default config', () => {
    const panel = new OliveConfigPanel();
    const el = (panel as any).element as HTMLElement;

    expect(el.className).toBe('demo-olive-config-panel');
    const quantSelect = el.querySelector('.demo-olive-quant-select') as HTMLSelectElement;
    expect(quantSelect).not.toBeNull();
    expect(quantSelect.value).toBe('None');

    const shapeCheckbox = el.querySelector('.demo-olive-shape-checkbox') as HTMLInputElement;
    expect(shapeCheckbox).not.toBeNull();
    expect(shapeCheckbox.checked).toBe(true);

    const fusionCheckbox = el.querySelector('.demo-olive-fusion-checkbox') as HTMLInputElement;
    expect(fusionCheckbox).not.toBeNull();
    expect(fusionCheckbox.checked).toBe(false);

    expect(panel.getConfig()).toEqual({
      quantizationLevel: 'None',
      enableStaticShapeInference: true,
      enableTransformerFusion: false
    });
  });

  it('should emit config on quant change', () => {
    const panel = new OliveConfigPanel();
    const el = (panel as any).element as HTMLElement;
    panel.mount(document.body);

    const spy = vi.fn();
    globalEventBus.on('OLIVE_CONFIG_CHANGED', spy);

    const quantSelect = el.querySelector('.demo-olive-quant-select') as HTMLSelectElement;
    quantSelect.value = 'INT8';
    quantSelect.dispatchEvent(new Event('change'));

    expect(spy).toHaveBeenCalledWith({
      quantizationLevel: 'INT8',
      enableStaticShapeInference: true,
      enableTransformerFusion: false
    });
    expect(panel.getConfig().quantizationLevel).toBe('INT8');

    panel.unmount();
  });

  it('should emit config on shape inference change', () => {
    const panel = new OliveConfigPanel();
    const el = (panel as any).element as HTMLElement;
    panel.mount(document.body);

    const spy = vi.fn();
    globalEventBus.on('OLIVE_CONFIG_CHANGED', spy);

    const shapeCheckbox = el.querySelector('.demo-olive-shape-checkbox') as HTMLInputElement;
    shapeCheckbox.checked = false;
    shapeCheckbox.dispatchEvent(new Event('change'));

    expect(spy).toHaveBeenCalledWith({
      quantizationLevel: 'None',
      enableStaticShapeInference: false,
      enableTransformerFusion: false
    });
    expect(panel.getConfig().enableStaticShapeInference).toBe(false);

    panel.unmount();
  });

  it('should emit config on transformer fusion change', () => {
    const panel = new OliveConfigPanel();
    const el = (panel as any).element as HTMLElement;
    panel.mount(document.body);

    const spy = vi.fn();
    globalEventBus.on('OLIVE_CONFIG_CHANGED', spy);

    const fusionCheckbox = el.querySelector('.demo-olive-fusion-checkbox') as HTMLInputElement;
    fusionCheckbox.checked = true;
    fusionCheckbox.dispatchEvent(new Event('change'));

    expect(spy).toHaveBeenCalledWith({
      quantizationLevel: 'None',
      enableStaticShapeInference: true,
      enableTransformerFusion: true
    });
    expect(panel.getConfig().enableTransformerFusion).toBe(true);

    panel.unmount();
  });
});
