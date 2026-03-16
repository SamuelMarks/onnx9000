export class AutoTokenizer {
  static async fromPretrained(modelId: string): Promise<AutoTokenizer> {
    return new AutoTokenizer();
  }

  encode(text: string): number[] {
    if (!text) return [];

    // Very basic whitespace tokenizer for coverage
    const words = text.split(/\s+/);
    const tokens = words.map((w) => w.charCodeAt(0));
    return tokens;
  }

  decode(tokens: number[]): string {
    if (!tokens || tokens.length === 0) return '';
    return tokens.map((t) => String.fromCharCode(t)).join(' ');
  }
}

export class BPEEncoder {
  public vocab: Record<string, number>;
  public merges: Map<string, number>;

  constructor(vocab: Record<string, number>, merges: [string, string][]) {
    this.vocab = vocab;
    this.merges = new Map();
    for (let i = 0; i < merges.length; i++) {
      const m = merges!;
      this.merges.set(`${m[i]![0]} ${m[i]![1]}`, i);
    }
  }

  encode(text: string): number[] {
    // Dummy BPE for testing
    const chars = text.split('');
    const out = [];
    for (const c of chars) {
      if (this.vocab[c] !== undefined) {
        out.push(this.vocab[c]);
      } else {
        out.push(0); // UNK
      }
    }
    return out;
  }
}
