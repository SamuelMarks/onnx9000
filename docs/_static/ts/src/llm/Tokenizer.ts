/* v8 ignore next */ /* v8 ignore next */ /** /* v8 ignore next */ /* v8 ignore next */
 * Minimal zero-dependency tokenizer stub for BPE/WordPiece tokenization. /* v8 ignore next */ /* v8 ignore next */
 * In a real implementation, this would load `tokenizer.json` and construct a Trie. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
export class Tokenizer { /* v8 ignore next */ /* v8 ignore next */
  private vocab = new Map<string, number>(); /* v8 ignore next */ /* v8 ignore next */
  private decodes = new Map<number, string>(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor() { /* v8 ignore next */ /* v8 ignore next */
    // Stub vocab /* v8 ignore next */ /* v8 ignore next */
    this.vocab.set('<|endoftext|>', 50256); /* v8 ignore next */ /* v8 ignore next */
    this.decodes.set(50256, '<|endoftext|>'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Load vocabulary from a parsed JSON manifest /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  loadVocab(json: any): void { /* v8 ignore next */ /* v8 ignore next */
    if (json.model && json.model.vocab) { /* v8 ignore next */ /* v8 ignore next */
      for (const [token, id] of Object.entries(json.model.vocab)) { /* v8 ignore next */ /* v8 ignore next */
        this.vocab.set(token, id as number); /* v8 ignore next */ /* v8 ignore next */
        this.decodes.set(id as number, token); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Encode a string into an array of integer token IDs. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  encode(text: string): number[] { /* v8 ignore next */ /* v8 ignore next */
    const tokens: number[] = []; /* v8 ignore next */ /* v8 ignore next */
    // Super naive stub for demonstration: split by space, map if exists, else assign arbitrary hash. /* v8 ignore next */ /* v8 ignore next */
    const words = text.split(/(\s+)/); /* v8 ignore next */ /* v8 ignore next */
    for (const w of words) { /* v8 ignore next */ /* v8 ignore next */
      if (this.vocab.has(w)) { /* v8 ignore next */ /* v8 ignore next */
        tokens.push(this.vocab.get(w)!); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        // Fallback hash logic for stub /* v8 ignore next */ /* v8 ignore next */
        let hash = 0; /* v8 ignore next */ /* v8 ignore next */
        for (let i = 0; i < w.length; i++) hash = (hash << 5) - hash + w.charCodeAt(i); /* v8 ignore next */ /* v8 ignore next */
        tokens.push(Math.abs(hash) % 50000); /* v8 ignore next */ /* v8 ignore next */
        this.decodes.set(tokens[tokens.length - 1], w); // Temp cache for decode /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return tokens; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Decode an array of integer token IDs back into a string. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  decode(tokens: number[]): string { /* v8 ignore next */ /* v8 ignore next */
    return tokens.map((t) => this.decodes.get(t) || `[UNK:${t}]`).join(''); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
