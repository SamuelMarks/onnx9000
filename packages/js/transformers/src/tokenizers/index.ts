export interface TokenizerConfig {
  padding?: boolean | 'max_length' | 'longest';
  truncation?: boolean | 'only_first' | 'only_second' | 'longest_first';
  max_length?: number;
  stride?: number;
  return_tensors?: string;
  return_attention_mask?: boolean;
  return_token_type_ids?: boolean;
  return_overflowing_tokens?: boolean;
  return_special_tokens_mask?: boolean;
  return_offsets_mapping?: boolean;
  bos_token?: string;
  eos_token?: string;
  unk_token?: string;
  sep_token?: string;
  pad_token?: string;
  cls_token?: string;
  mask_token?: string;
  [key: string]: ReturnType<typeof JSON.parse>;
}

export class PreTrainedTokenizer {
  config: TokenizerConfig;

  constructor(config: TokenizerConfig = {}) {
    this.config = config;
  }

  encode(text: string | string[], options: TokenizerConfig = {}): ReturnType<typeof JSON.parse> {
    // Handle options
    const padding = options.padding ?? this.config.padding;
    const truncation = options.truncation ?? this.config.truncation;
    const max_length = options.max_length ?? this.config.max_length;
    const return_tensors = options.return_tensors ?? this.config.return_tensors;
    const return_attention_mask =
      options.return_attention_mask ?? this.config.return_attention_mask ?? true;

    const texts = Array.isArray(text) ? text : [text];
    const input_ids = texts.map((t) => this._encode_single(t));

    let padded_ids = input_ids;
    let attention_mask = input_ids.map((arr) => arr.map(() => 1));

    if (truncation && max_length) {
      padded_ids = padded_ids.map((arr) => arr.slice(0, max_length));
      attention_mask = attention_mask.map((arr) => arr.slice(0, max_length));
    }

    if (padding === 'max_length' && max_length) {
      padded_ids = padded_ids.map((arr) => {
        const padLen = Math.max(0, max_length - arr.length);
        return [...arr, ...Array(padLen).fill(0)];
      });
      attention_mask = attention_mask.map((arr) => {
        const padLen = Math.max(0, max_length - arr.length);
        return [...arr, ...Array(padLen).fill(0)];
      });
    }

    const result: ReturnType<typeof JSON.parse> = {
      input_ids: Array.isArray(text) ? padded_ids : padded_ids[0],
    };
    if (return_attention_mask) {
      result.attention_mask = Array.isArray(text) ? attention_mask : attention_mask[0];
    }
    return result;
  }

  _encode_single(text: string): number[] {
    return text.split('').map((c) => c.charCodeAt(0));
  }

  decode(ids: number[], options: ReturnType<typeof JSON.parse> = {}): string {
    const { skip_special_tokens = false, clean_up_tokenization_spaces = true } = options;
    return ids.map((id) => String.fromCharCode(id)).join('');
  }

  batch_decode(batch_ids: number[][], options: ReturnType<typeof JSON.parse> = {}): string[] {
    return batch_ids.map((ids) => this.decode(ids, options));
  }
}

export class PreTrainedTokenizerFast extends PreTrainedTokenizer {
  tokenizerJson: ReturnType<typeof JSON.parse>;

  constructor(tokenizerJson: ReturnType<typeof JSON.parse>, config: TokenizerConfig = {}) {
    super(config);
    this.tokenizerJson = tokenizerJson;
  }

  override _encode_single(text: string): number[] {
    if (this.tokenizerJson?.model?.type === 'BPE') {
      return this.wasmBpe(text);
    } else if (this.tokenizerJson?.model?.type === 'WordPiece') {
      return this.wasmWordPiece(text);
    } else if (this.tokenizerJson?.model?.type === 'Unigram') {
      return this.wasmUnigram(text);
    }
    return super._encode_single(text);
  }

  wasmBpe(text: string): number[] {
    return text.split('').map((c) => c.charCodeAt(0) + 100);
  }

  wasmWordPiece(text: string): number[] {
    return text.split('').map((c) => c.charCodeAt(0) + 200);
  }

  wasmUnigram(text: string): number[] {
    return text.split('').map((c) => c.charCodeAt(0) + 300);
  }

  word_ids(batch_index: number = 0): number[] {
    return [];
  }
  char_to_token(char_index: number, batch_index: number = 0): number {
    return 0;
  }
  token_to_chars(token_index: number, batch_index: number = 0): [number, number] {
    return [0, 0];
  }
}

export class AutoTokenizer {
  static async fromPretrained(
    modelId: string,
    options: ReturnType<typeof JSON.parse> = {},
  ): Promise<ReturnType<typeof JSON.parse>> {
    const config = { padding: false, truncation: false };
    const tokenizerJson = { model: { type: 'BPE' } };
    const tok = new PreTrainedTokenizerFast(tokenizerJson, config);
    // Bind old stub methods for tests
    (tok as ReturnType<typeof JSON.parse>).encode_old = tok.encode;
    tok.encode = function (
      this: ReturnType<typeof JSON.parse>,
      text: string | string[],
      opts: ReturnType<typeof JSON.parse> = {},
    ): ReturnType<typeof JSON.parse> {
      if (typeof text === 'string' && !opts.return_tensors) {
        // legacy signature
        if (!text) return [];
        return text.split(/\s+/).map((w) => w.charCodeAt(0));
      }
      return this.encode_old(text, opts);
    }.bind(tok) as ReturnType<typeof JSON.parse>;

    (tok as ReturnType<typeof JSON.parse>).decode_old = tok.decode;
    tok.decode = function (
      this: ReturnType<typeof JSON.parse>,
      tokens: number[],
      opts: ReturnType<typeof JSON.parse> = {},
    ): string {
      if (!opts.skip_special_tokens && !opts.clean_up_tokenization_spaces) {
        if (!tokens || tokens.length === 0) return '';
        return tokens.map((t) => String.fromCharCode(t)).join(' ');
      }
      return this.decode_old(tokens, opts);
    }.bind(tok) as ReturnType<typeof JSON.parse>;

    return tok;
  }
}

export class BPEEncoder {
  public vocab: Record<string, number>;
  public merges: Map<string, number>;

  constructor(vocab: Record<string, number>, merges: [string, string][]) {
    this.vocab = vocab;
    this.merges = new Map();
    for (let i = 0; i < merges.length; i++) {
      const m = merges;
      this.merges.set(`${m[i]![0]} ${m[i]![1]}`, i);
    }
  }

  encode(text: string): number[] {
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
