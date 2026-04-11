/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TensorInputModal } from '../../src/components/TensorInputModal';
import { globalEventBus } from '../../src/core/EventBus';

// Mock URL.createObjectURL for JSDOM
global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
global.URL.revokeObjectURL = vi.fn();

describe('TensorInputModal', () => {
  beforeEach(() => {
    globalEventBus.clearAll();
  });

  it('should render hidden initially', () => {
    const modal = new TensorInputModal();
    const el = (modal as object).element as HTMLElement;

    expect(el.className).toBe('demo-tensor-modal-overlay');
    expect(el.style.display).toBe('none');
  });

  it('should show empty state if no inputs', () => {
    const modal = new TensorInputModal();
    const el = (modal as object).element as HTMLElement;

    modal.show([]);

    expect(el.style.display).toBe('flex');
    expect(el.querySelector('.demo-tensor-inputs-container')?.innerHTML).toContain(
      'No inputs required.'
    );
  });

  it('should show dynamically generated input fields', () => {
    const modal = new TensorInputModal();
    const el = (modal as object).element as HTMLElement;

    modal.show([{ name: 'image', type: 'float32', dims: ['N', 3, 224, 224] }]);

    const group = el.querySelector('.demo-tensor-input-group') as HTMLElement;
    expect(group).not.toBeNull();

    const label = group.querySelector('label') as HTMLElement;
    expect(label.innerHTML).toContain('image');
    expect(label.innerHTML).toContain('N, 3, 224, 224');

    const fileInput = group.querySelector('input[type="file"]') as HTMLInputElement;
    expect(fileInput).not.toBeNull();
    expect(fileInput.getAttribute('data-name')).toBe('image');
  });

  it('should close on closeBtn click or background click', () => {
    const modal = new TensorInputModal();
    const el = (modal as object).element as HTMLElement;
    modal.mount(document.body);

    modal.show([]);
    expect(el.style.display).toBe('flex');

    // Click close button
    const closeBtn = el.querySelector('.demo-btn-close') as HTMLButtonElement;
    closeBtn.click();
    expect(el.style.display).toBe('none');

    modal.show([]);

    // Click background
    el.dispatchEvent(new MouseEvent('click', { bubbles: true })); // Event target = el
    expect(el.style.display).toBe('none');

    // Clicking inner modal should NOT close
    modal.show([]);
    const innerModal = el.querySelector('.demo-tensor-modal') as HTMLElement;
    innerModal.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(el.style.display).toBe('flex');

    modal.unmount();
  });

  it('should emit toast on fillRandomData', () => {
    const modal = new TensorInputModal();
    const el = (modal as object).element as HTMLElement;
    modal.mount(document.body);

    const toastSpy = vi.fn();
    globalEventBus.on('SHOW_TOAST', toastSpy);

    const generateBtn = el.querySelector('.demo-btn-secondary') as HTMLButtonElement;
    generateBtn.click();

    expect(toastSpy).toHaveBeenCalledTimes(1);
    expect(toastSpy.mock.calls[0][0].message).toContain('Random data configured');

    modal.unmount();
  });

  it('should emit EXECUTE on submit and close', () => {
    const modal = new TensorInputModal();
    const el = (modal as object).element as HTMLElement;
    modal.mount(document.body);

    modal.show([]);

    const executeSpy = vi.fn();
    globalEventBus.on('EXECUTE_INFERENCE_REQUEST', executeSpy);

    const submitBtn = el.querySelector('.demo-btn-primary') as HTMLButtonElement;
    submitBtn.click();

    expect(executeSpy).toHaveBeenCalledTimes(1);
    expect(el.style.display).toBe('none');

    modal.unmount();
  });

  it('should handle file uploads and image Canvas resizing', () => {
    const modal = new TensorInputModal();
    const el = (modal as object).element as HTMLElement;
    modal.mount(document.body);

    modal.show([{ name: 'img', type: 'float32', dims: [1, 3, 256, 256] }]);

    const fileInput = el.querySelector('input[type="file"]') as HTMLInputElement;
    const canvas = el.querySelector('canvas') as HTMLCanvasElement;

    expect(canvas.style.display).toBe('none');

    // Mock an Image to load synchronously

    let onloadCallback: () => void = () => undefined;

    vi.spyOn(global, 'Image').mockImplementation(() => {
      return {
        set src(_val: object) {},
        set onload(cb: () => void) {
          onloadCallback = cb;
        }
      } as object as HTMLImageElement;
    });

    // Mock canvas context
    const drawImageSpy = vi.fn();
    canvas.getContext = vi.fn().mockReturnValue({ drawImage: drawImageSpy });

    // Mock File
    const file = new File([''], 'test.png', { type: 'image/png' });

    // Override file input target files
    Object.defineProperty(fileInput, 'files', {
      value: [file]
    });

    // Trigger change
    fileInput.dispatchEvent(new Event('change'));

    // Trigger onload
    onloadCallback();

    // Verify canvas dimensions were updated from dims (256, 256)
    expect(canvas.width).toBe(256);
    expect(canvas.height).toBe(256);

    // Verify draw image was called
    expect(drawImageSpy).toHaveBeenCalled();
    expect(canvas.style.display).toBe('block');

    modal.unmount();
  });

  it('should ignore file input change with no file', () => {
    const modal = new TensorInputModal();
    const el = (modal as object).element as HTMLElement;
    modal.mount(document.body);

    modal.show([{ name: 'img', type: 'float32', dims: [] }]);
    const fileInput = el.querySelector('input[type="file"]') as HTMLInputElement;

    Object.defineProperty(fileInput, 'files', { value: [] });

    expect(() => fileInput.dispatchEvent(new Event('change'))).not.toThrow();

    modal.unmount();
  });
});
