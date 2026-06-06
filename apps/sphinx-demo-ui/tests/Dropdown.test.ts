// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { Dropdown } from '../src/components/Dropdown.js';

describe('Dropdown', () => {
  it('should render and select', () => {
    const dropdown = new Dropdown({
      items: [
        { value: '1', label: 'One' },
        { value: '2', label: 'Two' }
      ],
      initialValue: '1'
    });
    document.body.appendChild(dropdown.element);

    expect(dropdown.getValue()).toBe('1');
    dropdown.toggle();
    dropdown.select('2');
    expect(dropdown.getValue()).toBe('2');
  });
});
