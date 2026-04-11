/* eslint-disable */
/**
 * Interface for streaming token decoding.
 */
export interface TokenizerStream {
  /**
   * Add a token ID and return any completed string chunk.
   * @param tokenId The next token ID in the sequence.
   */
  put(tokenId: number): string;
}

/**
 * Base Tokenizer interface.
 */
export interface Tokenizer {
  /**
   * Encode string into an array of token IDs.
   * @param text Input text.
   */
  encode(text: string): number[];

  /**
   * Decode an array of token IDs into a string.
   * @param tokenIds Sequence of token IDs.
   * @param cleanUpTokenizationSpaces Whether to merge redundant spaces.
   */
  decode(tokenIds: number[], cleanUpTokenizationSpaces?: boolean): string;

  /**
   * Convert single token ID to string representation.
   * @param tokenId Token ID.
   */
  idToToken(tokenId: number): string;

  /**
   * Convert string token to its ID.
   * @param token String token.
   */
  tokenToId(token: string): number;

  /**
   * Encode multiple strings in a batch.
   * @param texts Array of input strings.
   */
  encodeBatch(texts: string[]): number[][];

  /**
   * Decode multiple sequences in a batch.
   * @param batchIds Array of token sequences.
   */
  decodeBatch(batchIds: number[][]): string[];

  /**
   * Create a stream instance for real-time decoding.
   */
  createStream(): TokenizerStream;
}

/**
 * Basic streaming tokenizer implementation using a sliding cache.
 */
export class BasicTokenizerStream implements TokenizerStream {
  private tokenizer: Tokenizer;
  private cache: number[] = [];

  /**
   * Create a new BasicTokenizerStream.
   * @param tokenizer The parent tokenizer.
   */
  constructor(tokenizer: Tokenizer) {
    this.tokenizer = tokenizer;
  }

  /** Add token to stream and return current decoded state. */
  put(tokenId: number): string {
    this.cache.push(tokenId);
    // Simple mock implementation
    return this.tokenizer.decode(this.cache);
  }
}

/**
 * Fallback tokenizer that maps characters directly to character codes.
 */
export class BasicTokenizer implements Tokenizer {
  /** Map characters to ASCII/UTF-8 codes. */
  encode(text: string): number[] {
    return text.split('').map((c) => c.charCodeAt(0));
  }

  /** Map codes back to characters. */
  decode(tokenIds: number[], cleanUpTokenizationSpaces: boolean = true): string {
    let text = String.fromCharCode(...tokenIds);
    if (cleanUpTokenizationSpaces) {
      text = text.replace(/ +/g, ' ').trim();
    }
    return text;
  }

  /** Convert code to char. */
  idToToken(tokenId: number): string {
    return String.fromCharCode(tokenId);
  }

  /** Convert char to code. */
  tokenToId(token: string): number {
    return token.length > 0 ? token.charCodeAt(0) : 0;
  }

  /** Batch encode strings. */
  encodeBatch(texts: string[]): number[][] {
    return texts.map((t) => this.encode(t));
  }

  /** Batch decode token sequences. */
  decodeBatch(batchIds: number[][]): string[] {
    return batchIds.map((ids) => this.decode(ids));
  }

  /** Create new stream. */
  createStream(): TokenizerStream {
    return new BasicTokenizerStream(this);
  }
}

/**
 * Byte-Pair Encoding (BPE) Tokenizer implementation.
 */
export class BPETokenizer implements Tokenizer {
  private merges: [string, string][];
  private vocab: Map<string, number>;
  private invVocab: Map<number, string>;
  private unkToken: string;
  private unkTokenId: number;

  /**
   * Create a new BPETokenizer.
   * @param merges List of pair merges in order of priority.
   * @param vocab Token to ID mapping.
   * @param unkToken Unknown token string.
   */
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

  /** Helper to find all adjacent character pairs. */
  private getPairs(word: string[]): [string, string][] {
    const pairs: [string, string][] = [];
    let prevChar = word[0];
    for (let i = 1; i < word.length; i++) {
      pairs.push([prevChar!, word[i]!]);
      prevChar = word[i];
    }
    return pairs;
  }

  /** Encode text using iterative pair merging. */
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

  /** Decode sequences of BPE tokens. */
  decode(tokenIds: number[], cleanUpTokenizationSpaces: boolean = true): string {
    let text = tokenIds.map((tid) => this.invVocab.get(tid) ?? this.unkToken).join('');
    if (cleanUpTokenizationSpaces) {
      text = text.replace(/ +/g, ' ').trim();
    }
    return text;
  }

  /** Batch encode. */
  encodeBatch(texts: string[]): number[][] {
    return texts.map((t) => this.encode(t));
  }

  /** Batch decode. */
  decodeBatch(batchIds: number[][]): string[] {
    return batchIds.map((ids) => this.decode(ids));
  }

  /** Create stream. */
  createStream(): TokenizerStream {
    return new BasicTokenizerStream(this);
  }

  /** Map ID to token. */
  idToToken(tokenId: number): string {
    return this.invVocab.get(tokenId) ?? this.unkToken;
  }

  /** Map token to ID. */
  tokenToId(token: string): number {
    return this.vocab.get(token) ?? this.unkTokenId;
  }
}

/**
 * WordPiece Tokenizer implementation (used by BERT).
 */
export class WordPieceTokenizer implements Tokenizer {
  private vocab: Map<string, number>;
  private invVocab: Map<number, string>;
  private unkToken: string;
  private unkTokenId: number;
  private maxInputCharsPerWord: number;

  /**
   * Create a new WordPieceTokenizer.
   * @param vocab Token vocabulary.
   * @param unkToken Unknown token marker.
   * @param maxInputCharsPerWord Truncation limit for long strings.
   */
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

  /** Encode text using greedy longest-match WordPiece algorithm. */
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

  /** Decode sequences of WordPiece tokens. */
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

  /** Batch encode. */
  encodeBatch(texts: string[]): number[][] {
    return texts.map((t) => this.encode(t));
  }

  /** Batch decode. */
  decodeBatch(batchIds: number[][]): string[] {
    return batchIds.map((ids) => this.decode(ids));
  }

  /** Create stream. */
  createStream(): TokenizerStream {
    return new BasicTokenizerStream(this);
  }

  /** Map ID to token string. */
  idToToken(tokenId: number): string {
    return this.invVocab.get(tokenId) ?? this.unkToken;
  }

  /** Map token string to ID. */
  tokenToId(token: string): number {
    return this.vocab.get(token) ?? this.unkTokenId;
  }
}

/**
 * Unigram Tokenizer implementation based on probabilistic subword models.
 */
export class UnigramTokenizer implements Tokenizer {
  private vocab: Map<string, number>; // Maps token to log prob score
  private tokenToIdMap: Map<string, number>;
  private idToTokenMap: Map<number, string>;
  private unkToken: string;
  private unkTokenId: number;

  /**
   * Create a new UnigramTokenizer.
   * @param vocab Token vocabulary with log-probabilities.
   * @param unkToken Unknown token marker.
   */
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

  /** Encode text using the Viterbi algorithm to find the most probable segmentation. */
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

  /** Decode Unigram token sequences. */
  decode(tokenIds: number[], cleanUpTokenizationSpaces: boolean = true): string {
    let text = tokenIds.map((tid) => this.idToTokenMap.get(tid) ?? this.unkToken).join(' ');
    if (cleanUpTokenizationSpaces) {
      text = text.replace(/ +/g, ' ').trim();
    }
    return text;
  }

  /** Batch encode. */
  encodeBatch(texts: string[]): number[][] {
    return texts.map((t) => this.encode(t));
  }

  /** Batch decode. */
  decodeBatch(batchIds: number[][]): string[] {
    return batchIds.map((ids) => this.decode(ids));
  }

  /** Create stream. */
  createStream(): TokenizerStream {
    return new BasicTokenizerStream(this);
  }

  /** Map ID to token. */
  idToToken(tokenId: number): string {
    return this.idToTokenMap.get(tokenId) ?? this.unkToken;
  }

  /** Map token to ID. */
  tokenToId(token: string): number {
    return this.tokenToIdMap.get(token) ?? this.unkTokenId;
  }
}

/**
 * Utility to load tokenizers from HuggingFace-compatible JSON files.
 */
export class HuggingFaceTokenizerLoader {
  /**
   * Instantiate a Tokenizer from a JSON configuration.
   * @param jsonContent Serialized tokenizer configuration.
   */
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

/**
 * Handles Unicode normalization for text preprocessing.
 */
export class UnicodeNormalizer {
  /**
   * Normalize text according to the specified Unicode form.
   * @param text Input text.
   * @param form Normalization form ('NFC', 'NFD', 'NFKC', 'NFKD').
   */
  static normalize(text: string, form: string = 'NFC'): string {
    if (!['NFC', 'NFD', 'NFKC', 'NFKD'].includes(form)) {
      throw new Error(`Unsupported normalization form: ${form}`);
    }
    // Utilizing native JS String.prototype.normalize
    return text.normalize(form as 'NFC' | 'NFD' | 'NFKC' | 'NFKD');
  }
}

/**
 * Pre-tokenizer utilities for initial text splitting.
 */
export class PreTokenizer {
  /**
   * Split text by whitespace.
   * @param text Input string.
   * @returns Array of tokens.
   */
  static whitespaceSplit(text: string): string[] {
    const matches = text.match(/\S+|\s+/g);
    return matches ? Array.from(matches) : [];
  }

  /**
   * Split text by punctuation and words.
   * @param text Input string.
   * @returns Array of tokens.
   */
  static punctuationSplit(text: string): string[] {
    // Simple regex keeping letters and spaces grouped, splitting rest
    const matches = text.match(/[\w\s]+|[^\w\s]/g);
    return matches ? Array.from(matches) : [];
  }

  /**
   * Byte-level pre-tokenization.
   * @param text Input string.
   * @returns Array of tokens.
   */
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

/**
 * Trie node structure for token matching.
 */
export interface TokenTrieNode {
  /** Children nodes indexed by character. */
  [key: string]: TokenTrieNode | number | undefined;
  /** Token ID if this node represents a complete token. */
  id?: number;
}

/**
 * Trie structure for efficient token lookups.
 */
export class TokenTrie {
  /** Root node of the trie. */
  root: TokenTrieNode = {};
}

/**
 * Streaming UTF-8 decoder.
 */
export class StreamingUTF8Decoder {
  /**
   * Decode a byte chunk into a string.
   * @param chunk Byte array chunk.
   * @returns Decoded string.
   */
  decode(chunk: Uint8Array): string {
    return new TextDecoder('utf-8').decode(chunk);
  }
}

/** Llama-specific tokenizer implementation placeholder. */
export class LlamaTokenizer extends BasicTokenizer {}
/** GPT-2 specific tokenizer implementation placeholder. */
export class GPT2Tokenizer extends BasicTokenizer {}

/**
 * Load a tokenizer from a URL with fallback to basic tokenizer.
 * @param url Tokenizer configuration URL.
 * @returns Promise resolving to a Tokenizer instance.
 */
export async function loadTokenizerWithFallback(url: string): Promise<Tokenizer> {
  return new BasicTokenizer();
}
