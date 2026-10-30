// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initGraphSurgeonDemo } from '../src/main.js';

describe('demo-graphsurgeon', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="mutate-btn"></button><div id="surgeon-output"></div>';
  });

  it('should mutate graph and update output', () => {
    document.body.innerHTML = '';
    initGraphSurgeonDemo();

    document.body.innerHTML = '<button id="mutate-btn"></button><div id="surgeon-output"></div>';
    initGraphSurgeonDemo();

    const btn = document.getElementById('mutate-btn') as HTMLButtonElement;
    const out = document.getElementById('surgeon-output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Success! Graph structure modified.');
    expect(btn.disabled).toBe(false);
  });
});
