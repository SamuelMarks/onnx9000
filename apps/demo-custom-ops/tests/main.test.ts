// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initCustomOpsDemo } from '../src/main.js';

describe('demo-custom-ops', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<input id="op-name"/><button id="register-op"></button><div id="registry"></div>';
  });

  it('should register custom ops and append to registry DOM', () => {
    document.body.innerHTML = '';
    initCustomOpsDemo();

    document.body.innerHTML =
      '<input id="op-name" value="NoRegistryOp"/><button id="register-op"></button>';
    initCustomOpsDemo();
    document.getElementById('register-op')?.click();

    document.body.innerHTML =
      '<input id="op-name"/><button id="register-op"></button><div id="registry"></div>';
    initCustomOpsDemo();

    const input = document.getElementById('op-name') as HTMLInputElement;
    const btn = document.getElementById('register-op') as HTMLButtonElement;
    const registry = document.getElementById('registry') as HTMLElement;

    btn.click();
    expect(registry.children).toHaveLength(0);

    input.value = 'CustomGelu';
    btn.click();

    expect(registry.children).toHaveLength(1);
    expect(registry.children[0]?.textContent).toBe('CustomGelu');
    expect(input.value).toBe('');
  });
});
