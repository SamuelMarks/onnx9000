// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { Component } from '../src/core/Component.js';

class TestComp extends Component<HTMLDivElement> {
  protected render() {
    return document.createElement('div');
  }
}

describe('Component', () => {
  it('should mount and unmount', () => {
    const comp = new TestComp();
    document.body.appendChild(comp.element);
    expect(document.body.contains(comp.element)).toBe(true);

    comp.unmount();
    expect(document.body.contains(comp.element)).toBe(false);
  });
});
