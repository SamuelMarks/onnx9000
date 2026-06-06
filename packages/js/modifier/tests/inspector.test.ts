import { describe, expect, it } from 'vitest';
import { InitializerInspector } from '../src/components/initializers/inspector.js';

describe('InitializerInspector', () => {
  it('should render scalar', () => {
    const container = document.createElement('div');
    const mutator: any = {};
    const insp = new InitializerInspector(container, mutator);

    const tensor: any = {
      dtype: 'float32',
      shape: [1],
      data: new Uint8Array(4),
    };
    insp.render(tensor);
    expect(container.innerHTML).toContain('Scalar Editor');
  });
});
