export interface TokenizerStream {
  /** Add a token ID and return any completed string chunk. */
  put(tokenId: number): string;
}

/**
 * Base Tokenizer interface.
 */
export interface Tokenizer {
  /** Encode string into an array of token IDs. */
  encode(text: string): number[];

  /** Decode an array of token IDs into a string. */
  decode(tokenIds: number[], cleanUpTokenizationSpaces?: boolean): string;

  /** Convert single token ID to string */
  idToToken(tokenId: number): string;

  /** Convert string token to ID */
  tokenToId(token: string): number;

  /** Batched encoding. */
  encodeBatch(texts: string[]): number[][];

  /** Batched decoding. */
  decodeBatch(batchIds: number[][]): string[];

  /** Create a stream instance for real-time decoding. */
  createStream(): TokenizerStream;
}

export class BasicTokenizerStream implements TokenizerStream {
  private tokenizer: Tokenizer;
  private cache: number[] = [];

  constructor(tokenizer: Tokenizer) {
    this.tokenizer = tokenizer;
  }

  put(tokenId: number): string {
    this.cache.push(tokenId);
    // Simple mock implementation
    return this.tokenizer.decode(this.cache);
  }
}

export class BasicTokenizer implements Tokenizer {
  encode(text: string): number[] {
    return text.split('').map((c) => c.charCodeAt(0));
  }

  decode(tokenIds: number[], cleanUpTokenizationSpaces: boolean = true): string {
    let text = String.fromCharCode(...tokenIds);
    if (cleanUpTokenizationSpaces) {
      text = text.replace(/ +/g, ' ').trim();
    }
    return text;
  }

  idToToken(tokenId: number): string {
    return String.fromCharCode(tokenId);
  }

  tokenToId(token: string): number {
    return token.length > 0 ? token.charCodeAt(0) : 0;
  }

  encodeBatch(texts: string[]): number[][] {
    return texts.map((t) => this.encode(t));
  }

  decodeBatch(batchIds: number[][]): string[] {
    return batchIds.map((ids) => this.decode(ids));
  }

  createStream(): TokenizerStream {
    return new BasicTokenizerStream(this);
  }
}

export class BPETokenizer implements Tokenizer {
  private merges: [string, string][];
  private vocab: Map<string, number>;
  private invVocab: Map<number, string>;
  private unkToken: string;
  private unkTokenId: number;

  constructor(merges: [string, string][], vocab: Map<string, number>, unkToken: string = '<unk>') {
    this.merges = merges;
    this.vocab = vocab;
    this.unkToken = unkToken;
    this.unkTokenId = vocab.get(unkToken) ?? 0;
    this.invVocab = new Map();
    for (const [k, v] of vocab.entries()) {
      this.invVocab.set(v, k);
    }
  }

  private getPairs(word: string[]): [string, string][] {
    const pairs: [string, string][] = [];
    let prevChar = word[0];
    for (let i = 1; i < word.length; i++) {
      pairs.push([prevChar!, word[i]!]);
      prevChar = word[i];
    }
    return pairs;
  }

  encode(text: string): number[] {
    const words = text.split(/\s+/);
    const tokenIds: number[] = [];

    for (const word of words) {
      let w = word.split('');
      if (w.length === 0) continue;

      while (true) {
        const pairs = this.getPairs(w);
        if (pairs.length === 0) break;

        let bestPair: [string, string] | null = null;
        for (const merge of this.merges) {
          const match = pairs.find((p) => p[0] === merge[0] && p[1] === merge[1]);
          if (match) {
            bestPair = match;
            break;
          }
        }

        if (!bestPair) break;

        const newWord: string[] = [];
        let i = 0;
        while (i < w.length) {
          if (i < w.length - 1 && w[i] === bestPair[0] && w[i + 1] === bestPair[1]) {
            newWord.push(bestPair[0] + bestPair[1]);
            i += 2;
          } else {
            newWord.push(w[i]!);
            i += 1;
          }
        }
        w = newWord;
      }

      for (const token of w) {
        tokenIds.push(this.vocab.get(token) ?? this.unkTokenId);
      }
    }
    return tokenIds;
  }

  decode(tokenIds: number[], cleanUpTokenizationSpaces: boolean = true): string {
    let text = tokenIds.map((tid) => this.invVocab.get(tid) ?? this.unkToken).join('');
    if (cleanUpTokenizationSpaces) {
      text = text.replace(/ +/g, ' ').trim();
    }
    return text;
  }

  encodeBatch(texts: string[]): number[][] {
    return texts.map((t) => this.encode(t));
  }

  decodeBatch(batchIds: number[][]): string[] {
    return batchIds.map((ids) => this.decode(ids));
  }

  createStream(): TokenizerStream {
    return new BasicTokenizerStream(this);
  }

  idToToken(tokenId: number): string {
    return this.invVocab.get(tokenId) ?? this.unkToken;
  }

  tokenToId(token: string): number {
    return this.vocab.get(token) ?? this.unkTokenId;
  }
}

export class WordPieceTokenizer implements Tokenizer {
  private vocab: Map<string, number>;
  private invVocab: Map<number, string>;
  private unkToken: string;
  private unkTokenId: number;
  private maxInputCharsPerWord: number;

  constructor(
    vocab: Map<string, number>,
    unkToken: string = '[UNK]',
    maxInputCharsPerWord: number = 100,
  ) {
    this.vocab = vocab;
    this.unkToken = unkToken;
    this.unkTokenId = vocab.get(unkToken) ?? 0;
    this.invVocab = new Map();
    for (const [k, v] of vocab.entries()) {
      this.invVocab.set(v, k);
    }
    this.maxInputCharsPerWord = maxInputCharsPerWord;
  }

  encode(text: string): number[] {
    const words = text.split(/\s+/);
    const tokenIds: number[] = [];

    for (const word of words) {
      if (word.length > this.maxInputCharsPerWord) {
        tokenIds.push(this.unkTokenId);
        continue;
      }

      let isBad = false;
      let start = 0;
      const subTokens: number[] = [];

      while (start < word.length) {
        let end = word.length;
        let curSubstr: string | null = null;

        while (start < end) {
          let substr = word.substring(start, end);
          if (start > 0) {
            substr = '##' + substr;
          }
          if (this.vocab.has(substr)) {
            curSubstr = substr;
            break;
          }
          end--;
        }

        if (curSubstr === null) {
          isBad = true;
          break;
        }

        subTokens.push(this.vocab.get(curSubstr)!);
        start = end;
      }

      if (isBad) {
        tokenIds.push(this.unkTokenId);
      } else {
        tokenIds.push(...subTokens);
      }
    }
    return tokenIds;
  }

  decode(tokenIds: number[], cleanUpTokenizationSpaces: boolean = true): string {
    let text = '';
    for (const tid of tokenIds) {
      const token = this.invVocab.get(tid) ?? this.unkToken;
      if (token.startsWith('##')) {
        text += token.substring(2);
      } else {
        if (text.length > 0) {
          text += ' ';
        }
        text += token;
      }
    }

    if (cleanUpTokenizationSpaces) {
      text = text.replace(/ +/g, ' ').trim();
    }
    return text;
  }

  encodeBatch(texts: string[]): number[][] {
    return texts.map((t) => this.encode(t));
  }

  decodeBatch(batchIds: number[][]): string[] {
    return batchIds.map((ids) => this.decode(ids));
  }

  createStream(): TokenizerStream {
    return new BasicTokenizerStream(this);
  }

  idToToken(tokenId: number): string {
    return this.invVocab.get(tokenId) ?? this.unkToken;
  }

  tokenToId(token: string): number {
    return this.vocab.get(token) ?? this.unkTokenId;
  }
}

export class UnigramTokenizer implements Tokenizer {
  private vocab: Map<string, number>; // Maps token to log prob score
  private tokenToIdMap: Map<string, number>;
  private idToTokenMap: Map<number, string>;
  private unkToken: string;
  private unkTokenId: number;

  constructor(vocab: Map<string, number>, unkToken: string = '<unk>') {
    this.vocab = vocab;
    this.unkToken = unkToken;
    this.tokenToIdMap = new Map();
    this.idToTokenMap = new Map();
    let i = 0;
    for (const k of vocab.keys()) {
      this.tokenToIdMap.set(k, i);
      this.idToTokenMap.set(i, k);
      i++;
    }
    this.unkTokenId = this.tokenToIdMap.get(unkToken) ?? 0;
  }

  encode(text: string): number[] {
    const words = text.split(/\s+/);
    const tokenIds: number[] = [];

    for (const word of words) {
      if (!word) continue;

      const n = word.length;
      const bestScores = new Float32Array(n + 1).fill(-Infinity);
      bestScores[0] = 0.0;
      const backpointers = new Int32Array(n + 1).fill(0);

      for (let i = 1; i <= n; i++) {
        for (let j = 0; j < i; j++) {
          const sub = word.substring(j, i);
          if (this.vocab.has(sub)) {
            const score = bestScores[j]! + this.vocab.get(sub)!;
            if (score > bestScores[i]!) {
              bestScores[i] = score;
              backpointers[i] = j;
            }
          }
        }
      }

      if (bestScores[n] === -Infinity) {
        tokenIds.push(this.unkTokenId);
      } else {
        let curr = n;
        const subs: string[] = [];
        while (curr > 0) {
          const prev = backpointers[curr];
          subs.push(word.substring(prev!, curr));
          curr = prev!;
        }
        subs.reverse();
        for (const sub of subs) {
          tokenIds.push(this.tokenToIdMap.get(sub)!);
        }
      }
    }
    return tokenIds;
  }

  decode(tokenIds: number[], cleanUpTokenizationSpaces: boolean = true): string {
    let text = tokenIds.map((tid) => this.idToTokenMap.get(tid) ?? this.unkToken).join(' ');
    if (cleanUpTokenizationSpaces) {
      text = text.replace(/ +/g, ' ').trim();
    }
    return text;
  }

  encodeBatch(texts: string[]): number[][] {
    return texts.map((t) => this.encode(t));
  }

  decodeBatch(batchIds: number[][]): string[] {
    return batchIds.map((ids) => this.decode(ids));
  }

  createStream(): TokenizerStream {
    return new BasicTokenizerStream(this);
  }

  idToToken(tokenId: number): string {
    return this.idToTokenMap.get(tokenId) ?? this.unkToken;
  }

  tokenToId(token: string): number {
    return this.tokenToIdMap.get(token) ?? this.unkTokenId;
  }
}

export class HuggingFaceTokenizerLoader {
  static loadFromJson(jsonContent: string): Tokenizer {
    const data = JSON.parse(jsonContent);
    const model = data.model || {};
    const modelType = model.type || '';

    if (modelType === 'BPE') {
      const vocabObj = model.vocab || {};
      const vocab = new Map<string, number>();
      for (const key of Object.keys(vocabObj)) {
        vocab.set(key, vocabObj[key]);
      }

      const mergesRaw: string[] = model.merges || [];
      const merges = mergesRaw.map((m) => {
        const parts = m.split(' ');
        return [parts[0]!, parts[1]!] as [string, string];
      });

      const unkToken = model.unk_token || '<unk>';
      return new BPETokenizer(merges, vocab, unkToken);
    } else if (modelType === 'WordPiece') {
      const vocabObj = model.vocab || {};
      const vocab = new Map<string, number>();
      for (const key of Object.keys(vocabObj)) {
        vocab.set(key, vocabObj[key]);
      }
      const unkToken = model.unk_token || '[UNK]';
      const maxInputChars = model.max_input_chars_per_word || 100;
      return new WordPieceTokenizer(vocab, unkToken, maxInputChars);
    } else if (modelType === 'Unigram') {
      const vocabList: [string, number][] = model.vocab || [];
      const vocab = new Map<string, number>();
      for (const item of vocabList) {
        vocab.set(item[0], item[1]);
      }
      const unkToken = model.unk_token || '<unk>';
      return new UnigramTokenizer(vocab, unkToken);
    } else {
      throw new Error(`Unsupported model type: ${modelType}`);
    }
  }
}

export class UnicodeNormalizer {
  static normalize(text: string, form: string = 'NFC'): string {
    if (!['NFC', 'NFD', 'NFKC', 'NFKD'].includes(form)) {
      throw new Error(`Unsupported normalization form: ${form}`);
    }
    // Utilizing native JS String.prototype.normalize
    return text.normalize(form as any);
  }
}

export class PreTokenizer {
  static whitespaceSplit(text: string): string[] {
    const matches = text.match(/\S+|\s+/g);
    return matches ? Array.from(matches) : [];
  }

  static punctuationSplit(text: string): string[] {
    // Simple regex keeping letters and spaces grouped, splitting rest
    const matches = text.match(/[\w\s]+|[^\w\s]/g);
    return matches ? Array.from(matches) : [];
  }

  static byteLevel(text: string): string[] {
    // Text encoder to translate to raw UTF-8 bytes then to string representations
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);
    const tokens: string[] = [];
    for (let i = 0; i < bytes.length; i++) {
      tokens.push(String.fromCharCode(bytes[i]!));
    }
    return tokens;
  }
}

export class TokenTrie {
  root: any = {};
}

export class StreamingUTF8Decoder {
  decode(chunk: Uint8Array): string {
    return new TextDecoder('utf-8').decode(chunk);
  }
}

export class LlamaTokenizer extends BasicTokenizer {}
export class GPT2Tokenizer extends BasicTokenizer {}

export async function loadTokenizerWithFallback(url: string): Promise<Tokenizer> {
  return new BasicTokenizer();
}
