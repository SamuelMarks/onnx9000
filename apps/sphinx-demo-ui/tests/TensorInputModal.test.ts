// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { TensorInputModal } from '../src/components/TensorInputModal.js';

describe('TensorInputModal', () => {
  it('should render and show', () => {
    const modal = new TensorInputModal();
    document.body.appendChild(modal.element);

    expect(modal.element.style.display).toBe('none');

    modal.show([{ name: 'input', type: 'float32', dims: [1, 3, 224, 224] }]);
    expect(modal.element.style.display).toBe('flex');
    expect(modal.element.innerHTML).toContain('input');

    modal.hide();
    expect(modal.element.style.display).toBe('none');
  });
});
