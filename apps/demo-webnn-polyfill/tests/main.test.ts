import { describe, it, expect, vi } from 'vitest';
import { initWebnnDemo } from '../src/main.js';

describe('demo-webnn-polyfill', () => {
  it('should run webnn demo', async () => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="webnn-output"></div>';

    // Mock WebNN
    (global.navigator as any).ml = {
      createContext: vi.fn().mockResolvedValue({
        compute: vi.fn().mockResolvedValue({ outputs: { y: [1, 2] } }),
      }),
    };
    (global as any).MLGraphBuilder = class {
      constructor() {}
      input() {
        return {};
      }
      constant() {
        return {};
      }
      matmul() {
        return {};
      }
      add() {
        return {};
      }
      build() {
        return {};
      }
    };

    initWebnnDemo();
    document.getElementById('run-btn')?.click();
    await new Promise((r) => setTimeout(r, 10));
    expect(document.getElementById('webnn-output')?.textContent).toContain('Success!');
  });
});
