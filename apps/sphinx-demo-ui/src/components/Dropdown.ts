import { Component } from '../core/Component';

export interface DropdownItem {
  value: string;
  label: string;
  group?: string;
}

export interface DropdownOptions {
  items: DropdownItem[];
  initialValue?: string;
  placeholder?: string;
  onChange?: (value: string) => void;
}

/**
 * A highly accessible, custom vanilla JS dropdown component.
 * Supports keyboard navigation (Arrow Up/Down, Enter, Escape).
 */
export class Dropdown extends Component<HTMLDivElement> {
  private options: DropdownOptions;
  private selectedValue: string | null = null;
  private isOpen = false;
  private focusedIndex = -1;

  private button!: HTMLButtonElement;
  private listbox!: HTMLUListElement;

  constructor(options: DropdownOptions) {
    super();
    this.options = { placeholder: 'Select an option...', ...options };

    if (this.options.initialValue) {
      this.selectedValue = this.options.initialValue;
    }

    this.element = this.render();
    this.updateButtonText();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-dropdown';
    container.setAttribute('aria-expanded', 'false');
    container.setAttribute('aria-haspopup', 'listbox');

    this.button = document.createElement('button');
    this.button.className = 'demo-dropdown-button';
    this.button.setAttribute('type', 'button');
    this.button.setAttribute('aria-labelledby', 'dropdown-button-label');

    // Add arrow icon or indicator inside button if needed via CSS pseudo-elements

    this.listbox = document.createElement('ul');
    this.listbox.className = 'demo-dropdown-listbox';
    this.listbox.setAttribute('role', 'listbox');
    this.listbox.style.display = 'none';

    this.renderItems();

    container.appendChild(this.button);
    container.appendChild(this.listbox);

    return container;
  }

  private renderItems(): void {
    this.listbox.innerHTML = ''; // Clear
    this.options.items.forEach((item, index) => {
      const li = document.createElement('li');
      li.className = 'demo-dropdown-item';
      li.setAttribute('role', 'option');
      li.setAttribute('data-value', item.value);
      li.setAttribute('data-index', index.toString());
      li.textContent = item.label;

      if (item.value === this.selectedValue) {
        li.setAttribute('aria-selected', 'true');
        li.classList.add('selected');
      }

      this.listbox.appendChild(li);
    });
  }

  protected onMount(): void {
    // Click toggle
    this.addDOMListener(this.button, 'click', () => this.toggle());

    // Click outside to close
    this.addDOMListener(document, 'mousedown', (e) => {
      if (!this.element.contains(e.target as Node)) {
        this.close();
      }
    });

    // Item click
    this.addDOMListener(this.listbox, 'mousedown', (e) => {
      const target = e.target as HTMLElement;
      const li = target.closest('.demo-dropdown-item') as HTMLLIElement | null;
      if (li) {
        const val = li.getAttribute('data-value');
        if (val) {
          this.select(val);
        }
      }
    });

    // Keyboard Navigation
    this.addDOMListener(this.element, 'keydown', (e) => this.handleKeyDown(e as KeyboardEvent));
  }

  private handleKeyDown(e: KeyboardEvent): void {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (!this.isOpen) {
          this.open();
        } else {
          this.focusNext();
        }
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (this.isOpen) {
          this.focusPrev();
        }
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        if (this.isOpen && this.focusedIndex >= 0) {
          const item = this.options.items[this.focusedIndex];
          if (item) {
            this.select(item.value);
          }
        } else {
          this.toggle();
        }
        break;
      case 'Escape':
        if (this.isOpen) {
          e.preventDefault();
          this.close();
          this.button.focus();
        }
        break;
    }
  }

  private focusNext(): void {
    this.focusedIndex = (this.focusedIndex + 1) % this.options.items.length;
    this.updateFocus();
  }

  private focusPrev(): void {
    this.focusedIndex = this.focusedIndex - 1;
    if (this.focusedIndex < 0) {
      this.focusedIndex = this.options.items.length - 1;
    }
    this.updateFocus();
  }

  private updateFocus(): void {
    const items = this.listbox.querySelectorAll('.demo-dropdown-item');
    items.forEach((item, index) => {
      if (index === this.focusedIndex) {
        item.classList.add('focused');
        if (typeof (item as HTMLElement).scrollIntoView === 'function')
          (item as HTMLElement).scrollIntoView({ block: 'nearest' });
      } else {
        item.classList.remove('focused');
      }
    });
  }

  public toggle(): void {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }

  public open(): void {
    this.isOpen = true;
    this.listbox.style.display = 'block';
    this.element.setAttribute('aria-expanded', 'true');

    // Reset focus to selected item or first item
    const selectedIdx = this.options.items.findIndex((i) => i.value === this.selectedValue);
    this.focusedIndex = selectedIdx >= 0 ? selectedIdx : 0;
    this.updateFocus();
  }

  public close(): void {
    this.isOpen = false;
    this.listbox.style.display = 'none';
    this.element.setAttribute('aria-expanded', 'false');
    this.focusedIndex = -1;

    // Remove focused classes
    const items = this.listbox.querySelectorAll('.demo-dropdown-item.focused');
    items.forEach((i) => i.classList.remove('focused'));
  }

  public select(value: string): void {
    this.selectedValue = value;
    this.updateButtonText();

    // Re-render items to update aria-selected attributes
    this.renderItems();
    this.close();

    if (this.options.onChange) {
      this.options.onChange(value);
    }
  }

  public updateItems(newItems: DropdownItem[]): void {
    this.options.items = newItems;

    // Ensure selected value still exists, else clear it
    if (this.selectedValue && !newItems.find((i) => i.value === this.selectedValue)) {
      this.selectedValue = null;
    }

    this.renderItems();
    this.updateButtonText();
  }

  private updateButtonText(): void {
    const selectedItem = this.options.items.find((i) => i.value === this.selectedValue);
    this.button.textContent = selectedItem ? selectedItem.label : this.options.placeholder || '';
  }

  public getValue(): string | null {
    return this.selectedValue;
  }
}
