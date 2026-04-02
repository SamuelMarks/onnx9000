/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import '../src/ui.js';

describe('TritonCompilerElement', () => {
  let el: any;

  beforeEach(() => {
    document.body.innerHTML = '';
    el = document.createElement('triton-compiler-ui');
    document.body.appendChild(el);
  });

  it('should handle drag events', () => {
    const dropzone = el.shadowRoot.querySelector('#dropzone');

    const dragOver = new Event('dragover');
    dragOver.preventDefault = vi.fn();
    dropzone.dispatchEvent(dragOver);
    expect(dropzone.classList.contains('hover')).toBe(true);

    const dragLeave = new Event('dragleave');
    dropzone.dispatchEvent(dragLeave);
    expect(dropzone.classList.contains('hover')).toBe(false);
  });

  it('should handle all slider inputs', () => {
    ['#blockM', '#blockN', '#blockK'].forEach((id, idx) => {
      const valId = ['#bm-val', '#bn-val', '#bk-val'][idx];
      const slider = el.shadowRoot.querySelector(id);
      const val = el.shadowRoot.querySelector(valId);
      slider.value = '128';
      slider.dispatchEvent(new Event('input'));
      expect(val.textContent).toBe('128');
    });
  });

  it('should handle generate button click', () => {
    const spy = vi.fn();
    el.addEventListener('generate-requested', spy);
    el.shadowRoot.querySelector('#gen').click();
    expect(spy).toHaveBeenCalled();
  });

  it('should handle file drop', () => {
    const dropzone = el.shadowRoot.querySelector('#dropzone');
    const spy = vi.fn();
    el.addEventListener('model-loaded', spy);

    const event = new Event('drop') as any;
    event.dataTransfer = {
      files: [{ name: 'model.onnx' }],
    };
    event.preventDefault = vi.fn();
    dropzone.dispatchEvent(event);

    expect(spy).toHaveBeenCalled();
    expect(dropzone.innerHTML).toContain('Loaded: model.onnx');
  });

  it('should support bundle method', () => {
    const graph = { inputs: [{ name: 'scale', shape: [] }] };
    const code = el.bundle(graph);
    expect(code).toContain('Uniforms: scale');
  });

  it('should set code in editors', () => {
    el.setCode('print(\"hi\")', 'compute main()');
    expect(el.shadowRoot.querySelector('#output').textContent).toBe('print(\"hi\")');
    expect(el.shadowRoot.querySelector('#wgsl-output').textContent).toBe('compute main()');
  });

  it('should handle save button click', () => {
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:test');
    const aClickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    el.shadowRoot.querySelector('#save').click();
    expect(aClickSpy).toHaveBeenCalled();
    aClickSpy.mockRestore();
  });
});
