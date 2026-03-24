import { describe, it, expect, vi } from 'vitest';
import { Component } from '../../src/core/Component';

class TestComponent extends Component<HTMLDivElement> {
  constructor() {
    super();
    this.element = this.render();
  }
  protected render(): HTMLDivElement {
    const div = document.createElement('div');
    div.id = 'test-div';
    div.textContent = 'Hello World';
    return div;
  }

  public exposeAddDOMListener(
    target: EventTarget,
    event: string,
    handler: EventListenerOrEventListenerObject
  ) {
    this.addDOMListener(target, event, handler);
  }

  public exposeOnCleanup(fn: () => void) {
    this.onCleanup(fn);
  }
}

describe('Component', () => {
  it('should create an element upon instantiation', () => {
    const component = new TestComponent();
    // Use an any cast to access protected property for testing purposes
    expect((component as any).element).toBeInstanceOf(HTMLDivElement);
    expect((component as any).element.id).toBe('test-div');
  });

  it('should mount correctly into a parent element', () => {
    const component = new TestComponent();
    const parent = document.createElement('div');

    // Create a spy for onMount
    const onMountSpy = vi.spyOn(component as any, 'onMount');

    component.mount(parent);

    expect(parent.children.length).toBe(1);
    expect(parent.children[0].id).toBe('test-div');
    expect(onMountSpy).toHaveBeenCalledTimes(1);
  });

  it('should replace an existing DOM element', () => {
    const component = new TestComponent();
    const parent = document.createElement('div');
    const oldChild = document.createElement('span');
    parent.appendChild(oldChild);

    const onMountSpy = vi.spyOn(component as any, 'onMount');

    component.replace(oldChild);

    expect(parent.children.length).toBe(1);
    expect(parent.children[0].id).toBe('test-div');
    expect(onMountSpy).toHaveBeenCalledTimes(1);
  });

  it('should safely do nothing when replace is called on element without parent', () => {
    const component = new TestComponent();
    const floatingElement = document.createElement('span');

    expect(() => component.replace(floatingElement)).not.toThrow();
  });

  it('should unmount and remove itself from the DOM', () => {
    const component = new TestComponent();
    const parent = document.createElement('div');
    component.mount(parent);

    expect(parent.children.length).toBe(1);

    component.unmount();

    expect(parent.children.length).toBe(0);
  });

  it('should run cleanup functions on unmount', () => {
    const component = new TestComponent();
    const cleanupSpy = vi.fn();

    component.exposeOnCleanup(cleanupSpy);
    component.unmount();

    expect(cleanupSpy).toHaveBeenCalledTimes(1);
  });

  it('should remove DOM listeners on unmount', () => {
    const component = new TestComponent();
    const target = document.createElement('button');
    const handlerSpy = vi.fn();

    component.exposeAddDOMListener(target, 'click', handlerSpy);

    // Trigger click
    target.dispatchEvent(new Event('click'));
    expect(handlerSpy).toHaveBeenCalledTimes(1);

    // Unmount
    component.unmount();

    // Trigger click again
    target.dispatchEvent(new Event('click'));
    expect(handlerSpy).toHaveBeenCalledTimes(1); // Should not have increased
  });
});

class BadComponent extends Component<HTMLDivElement> {
  protected render(): HTMLDivElement {
    return document.createElement('div');
  }
}

describe('Component - Uninitialized Element', () => {
  it('should throw an error when mounting without an initialized element', () => {
    const bad = new BadComponent();
    expect(() => bad.mount(document.body)).toThrow('Component element not initialized');
  });

  it('should throw an error when replacing without an initialized element', () => {
    const bad = new BadComponent();
    const span = document.createElement('span');
    expect(() => bad.replace(span)).toThrow('Component element not initialized');
  });
});
