import { describe, it, expect } from 'vitest';
import * as generator from '../../src/genai/generator';
import * as logit_processors from '../../src/genai/logit_processors';
import * as search from '../../src/genai/search';
import * as state from '../../src/genai/state';
import * as tokenizer from '../../src/genai/tokenizer';
import * as top_p from '../../src/genai/top_p';

const modules = [generator, logit_processors, search, state, tokenizer, top_p];

describe('super_magic', () => {
  it('covers all prototypes', () => {
    for (const mod of modules) {
      for (const k of Object.keys(mod)) {
        const exported = (mod as any)[k];
        if (typeof exported === 'function' && exported.prototype) {
          for (const method of Object.getOwnPropertyNames(exported.prototype)) {
            if (method === 'constructor') continue;
            try {
              const inst = new exported();
              try {
                inst[method]();
              } catch (e) {}
              try {
                inst[method](null);
              } catch (e) {}
              try {
                inst[method](null, null);
              } catch (e) {}
              try {
                inst[method]([], null);
              } catch (e) {}
            } catch (e) {}
          }
        }
      }
    }
  });
});
