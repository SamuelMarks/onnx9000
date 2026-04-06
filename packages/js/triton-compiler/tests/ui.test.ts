/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import '../src/ui.js';
import type { TritonCompilerElement } from '../src/ui.js';

describe('TritonCompilerElement', () => {
  let el: TritonCompilerElement;

  beforeEach(() => {
    document.body.innerHTML = '';
    el = document.createElement('triton-compiler-ui') as TritonCompilerElement;
    document.body.appendChild(el);
  });

  it('should handle drag events', () => {
    const dropzone = el.shadowRoot!.querySelector('#dropzone') as HTMLElement;

    const dragOver = new Event('dragover') as Event & { preventDefault: () => void };
    dragOver.preventDefault = vi.fn();
    dropzone.dispatchEvent(dragOver);
    expect(dropzone.classList.contains('hover')).toBe(true);

    const dragLeave = new Event('dragleave');
    dropzone.dispatchEvent(dragLeave);
    expect(dropzone.classList.contains('hover')).toBe(false);
  });

  it('should handle all slider inputs', () => {
    ['#blockM', '#blockN', '#blockK'].forEach((id, idx) => {
      const valId = ['#bm-val', '#bn-val', '#bk-val'][idx] as string;
      const slider = el.shadowRoot!.querySelector(id) as HTMLInputElement;
      const val = el.shadowRoot!.querySelector(valId) as HTMLElement;
      slider.value = '128';
      slider.dispatchEvent(new Event('input'));
      expect(val.textContent).toBe('128');
    });
  });

  it('should handle generate button click', () => {
    const spy = vi.fn();
    el.addEventListener('generate-requested', spy);
    (el.shadowRoot!.querySelector('#gen') as HTMLElement).click();
    expect(spy).toHaveBeenCalled();
  });

  it('should handle file drop', () => {
    const dropzone = el.shadowRoot!.querySelector('#dropzone') as HTMLElement;
    const spy = vi.fn();
    el.addEventListener('model-loaded', spy);

    const event = new Event('drop') as Event & {
      dataTransfer: { files: { name: string }[] };
      preventDefault: () => void;
    };
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
    el.setCode('print("hi")', 'compute main()');
    expect((el.shadowRoot!.querySelector('#output') as HTMLElement).textContent).toBe(
      'print("hi")',
    );
    expect((el.shadowRoot!.querySelector('#wgsl-output') as HTMLElement).textContent).toBe(
      'compute main()',
    );
  });

  it('should handle save button click', () => {
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:test');
    const aClickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    (el.shadowRoot!.querySelector('#save') as HTMLElement).click();
    expect(aClickSpy).toHaveBeenCalled();
    aClickSpy.mockRestore();
  });
});
