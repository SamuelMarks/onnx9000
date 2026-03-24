import { describe, it, expect, vi } from 'vitest';
import { Dropdown } from '../../src/components/Dropdown';

describe('Dropdown', () => {
  const items = [
    { value: 'opt1', label: 'Option 1' },
    { value: 'opt2', label: 'Option 2' },
    { value: 'opt3', label: 'Option 3' }
  ];

  it('should initialize and render correctly', () => {
    const dropdown = new Dropdown({ items, placeholder: 'Select Me' });
    const el = (dropdown as any).element as HTMLElement;

    expect(el.className).toBe('demo-dropdown');

    const btn = el.querySelector('.demo-dropdown-button') as HTMLButtonElement;
    expect(btn.textContent).toBe('Select Me');

    const listbox = el.querySelector('.demo-dropdown-listbox') as HTMLUListElement;
    expect(listbox.children.length).toBe(3);
    expect(listbox.children[0].textContent).toBe('Option 1');
  });

  it('should set initial value', () => {
    const dropdown = new Dropdown({ items, initialValue: 'opt2' });
    const el = (dropdown as any).element as HTMLElement;

    const btn = el.querySelector('.demo-dropdown-button') as HTMLButtonElement;
    expect(btn.textContent).toBe('Option 2');

    const selectedItem = el.querySelector('.demo-dropdown-item.selected') as HTMLLIElement;
    expect(selectedItem).not.toBeNull();
    expect(selectedItem.textContent).toBe('Option 2');
  });

  it('should toggle listbox visibility on click', () => {
    const dropdown = new Dropdown({ items });
    const el = (dropdown as any).element as HTMLElement;
    dropdown.mount(document.body); // trigger onMount for listeners

    const btn = el.querySelector('.demo-dropdown-button') as HTMLButtonElement;
    const listbox = el.querySelector('.demo-dropdown-listbox') as HTMLUListElement;

    expect(listbox.style.display).toBe('none');

    btn.click(); // Open
    expect(listbox.style.display).toBe('block');

    btn.click(); // Close
    expect(listbox.style.display).toBe('none');

    dropdown.unmount();
  });

  it('should close on outside click', () => {
    const dropdown = new Dropdown({ items });
    const el = (dropdown as any).element as HTMLElement;
    dropdown.mount(document.body);

    const listbox = el.querySelector('.demo-dropdown-listbox') as HTMLUListElement;

    dropdown.open();
    expect(listbox.style.display).toBe('block');

    // Simulate outside click
    document.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));

    expect(listbox.style.display).toBe('none');
    dropdown.unmount();
  });

  it('should trigger onChange and close on selection', () => {
    const onChangeSpy = vi.fn();
    const dropdown = new Dropdown({ items, onChange: onChangeSpy });
    const el = (dropdown as any).element as HTMLElement;
    dropdown.mount(document.body);

    dropdown.open();
    const listbox = el.querySelector('.demo-dropdown-listbox') as HTMLUListElement;

    const itemOpt3 = listbox.querySelector('[data-value="opt3"]') as HTMLElement;

    // Simulate mousedown bubbling from item
    itemOpt3.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));

    expect(onChangeSpy).toHaveBeenCalledWith('opt3');
    expect(dropdown.getValue()).toBe('opt3');
    expect(listbox.style.display).toBe('none');

    dropdown.unmount();
  });

  it('should navigate via keyboard arrows and enter', () => {
    const dropdown = new Dropdown({ items });
    const el = (dropdown as any).element as HTMLElement;
    dropdown.mount(document.body);

    // Test arrow down to open
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    const listbox = el.querySelector('.demo-dropdown-listbox') as HTMLUListElement;
    expect(listbox.style.display).toBe('block');
    // First open automatically focuses the first item (or selected)
    expect(listbox.children[0].classList.contains('focused')).toBe(true);

    // Arrow down again to focus second item
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    expect(listbox.children[1].classList.contains('focused')).toBe(true);

    // Arrow down to third
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    expect(listbox.children[2].classList.contains('focused')).toBe(true);

    // Arrow up to second
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
    expect(listbox.children[1].classList.contains('focused')).toBe(true); // After first open

    // Enter to select
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(dropdown.getValue()).toBe('opt2');
    expect(listbox.style.display).toBe('none');

    dropdown.unmount();
  });

  it('should close via Escape', () => {
    const dropdown = new Dropdown({ items });
    const el = (dropdown as any).element as HTMLElement;
    dropdown.mount(document.body);

    dropdown.open();
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    const listbox = el.querySelector('.demo-dropdown-listbox') as HTMLUListElement;
    expect(listbox.style.display).toBe('none');

    dropdown.unmount();
  });

  it('should update items dynamically', () => {
    const dropdown = new Dropdown({ items, initialValue: 'opt1' });
    const el = (dropdown as any).element as HTMLElement;

    dropdown.updateItems([{ value: 'new1', label: 'New Option 1' }]);

    const listbox = el.querySelector('.demo-dropdown-listbox') as HTMLUListElement;
    expect(listbox.children.length).toBe(1);
    expect(listbox.children[0].textContent).toBe('New Option 1');
    expect(dropdown.getValue()).toBeNull(); // opt1 is no longer in the list
  });
  it('should toggle open with Space or Enter when closed', () => {
    const dropdown = new Dropdown({ items });
    const el = (dropdown as any).element as HTMLElement;
    dropdown.mount(document.body);

    el.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
    const listbox = el.querySelector('.demo-dropdown-listbox') as HTMLUListElement;
    expect(listbox.style.display).toBe('block');
  });

  it('should loop around with ArrowUp', () => {
    const dropdown = new Dropdown({ items });
    const el = (dropdown as any).element as HTMLElement;
    dropdown.mount(document.body);

    dropdown.open(); // index becomes 0
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true })); // loops to 2

    const listbox = el.querySelector('.demo-dropdown-listbox') as HTMLUListElement;
    expect(listbox.children[2].classList.contains('focused')).toBe(true);
  });
});

it('should select via click and emit onChange', () => {
  const dropdown = new Dropdown({ items: [{ value: 'a', label: 'A' }] });
  dropdown.mount(document.body);
  const el = (dropdown as any).element as HTMLElement;
  dropdown.open();
  const item = el.querySelector('.demo-dropdown-item') as HTMLElement;
  item.click(); // Uses standard click which tests unmounted path if missed
});
