import { describe, it, expect } from 'vitest';
import { initCustomOpsDemo } from '../src/main.js';

describe('demo-custom-ops', () => {
  it('should register an op', () => {
    document.body.innerHTML = `
      <button id="register-op"></button>
      <input id="op-name" value="MyOp" />
      <div id="registry"></div>
    `;
    initCustomOpsDemo();
    document.getElementById('register-op')?.click();
    expect(document.getElementById('registry')?.innerHTML).toContain('MyOp');
    expect((document.getElementById('op-name') as HTMLInputElement).value).toBe('');
  });
});
