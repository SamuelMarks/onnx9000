/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { Editor } from '../../src/components/Editor';

describe('LocalStorage Error Handling', () => {
  it('should gracefully handle localStorage parse/quota errors', () => {
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceededError');
    });
    const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    const editor = new Editor({ language: 'python', onChange: vi.fn() });
    editor.mount(document.body);

    const monacoInstance = (editor as object).editor;
    const registeredCallback = monacoInstance.onDidChangeModelContent.mock.calls[0][0];

    expect(() => registeredCallback()).not.toThrow();
    expect(consoleWarnSpy).toHaveBeenCalledWith(
      'Failed to save editor state to localStorage',
      expect.any(Error)
    );

    editor.unmount();
    setItemSpy.mockRestore();
    consoleWarnSpy.mockRestore();
  });
});
