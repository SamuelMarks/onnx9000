export abstract class Tokenizer {
    public vocab: Map<string, number>;
    public inverseVocab: Map<number, string>;
    public unkToken: string;
    public padToken: string;
    public clsToken: string;
    public sepToken: string;
    public maskToken: string;

    public unkId: number;
    public padId: number;
    public clsId: number;
    public sepId: number;
    public maskId: number;

    /** Implementation details and semantic operations. */

    constructor(
        vocab: Record<string, number>,
        unkToken: string = "[UNK]",
        padToken: string = "[PAD]",
        clsToken: string = "[CLS]",
        sepToken: string = "[SEP]",
        maskToken: string = "[MASK]"
    ) {
        this.vocab = new Map(Object.entries(vocab));
        this.inverseVocab = new Map();
        /** Implementation details and semantic operations. */
        for(const [k, v] of this.vocab.entries()) {
            this.inverseVocab.set(v, k);
        }

        this.unkToken = unkToken;
        this.padToken = padToken;
        this.clsToken = clsToken;
        this.sepToken = sepToken;
        this.maskToken = maskToken;

        this.unkId = this.vocab.get(this.unkToken) ?? -1;
        this.padId = this.vocab.get(this.padToken) ?? -1;
        this.clsId = this.vocab.get(this.clsToken) ?? -1;
        this.sepId = this.vocab.get(this.sepToken) ?? -1;
        this.maskId = this.vocab.get(this.maskToken) ?? -1;
    }

    abstract encode(text: string): number[];
    abstract decode(ids: number[]): string;

    /** Implementation details and semantic operations. */

    public encodePlus(
        text: string,
        textPair: string | null = null,
        maxLength: number | null = null,
        padToMaxLength: boolean = false,
        truncation: boolean = false
    ): { input_ids: number[]; attention_mask: number[]; token_type_ids: number[] } {
        const ids1 = this.encode(text);
        const ids2 = textPair ? this.encode(textPair) : [];

        let input_ids: number[] = [];
        let token_type_ids: number[] = [];

        /** Implementation details and semantic operations. */

        if(this.clsId !== -1) {
            input_ids.push(this.clsId);
            token_type_ids.push(0);
        }

        input_ids.push(...ids1);
        token_type_ids.push(...Array(ids1.length).fill(0));

        /** Implementation details and semantic operations. */

        if(this.sepId !== -1) {
            input_ids.push(this.sepId);
            token_type_ids.push(0);
        }

        /** Implementation details and semantic operations. */

        if(ids2.length > 0) {
            input_ids.push(...ids2);
            token_type_ids.push(...Array(ids2.length).fill(1));
            /** Implementation details and semantic operations. */
            if(this.sepId !== -1) {
                input_ids.push(this.sepId);
                token_type_ids.push(1);
            }
        }

        /** Implementation details and semantic operations. */

        if(truncation && maxLength !== null && input_ids.length > maxLength) {
            input_ids = input_ids.slice(0, maxLength);
            token_type_ids = token_type_ids.slice(0, maxLength);
            /** Implementation details and semantic operations. */
            if(this.sepId !== -1 && input_ids[input_ids.length - 1] !== this.sepId) {
                input_ids[input_ids.length - 1] = this.sepId;
            }
        }

        const attention_mask = Array(input_ids.length).fill(1);

        /** Implementation details and semantic operations. */

        if(padToMaxLength && maxLength !== null && input_ids.length < maxLength) {
            const padLen = maxLength - input_ids.length;
            input_ids.push(...Array(padLen).fill(this.padId));
            attention_mask.push(...Array(padLen).fill(0));
            token_type_ids.push(...Array(padLen).fill(0));
        }

        return {
            input_ids,
            attention_mask,
            token_type_ids,
        };
    }

    /** Implementation details and semantic operations. */

    public normalize(text: string, lower: boolean = true, stripAccents: boolean = true): string {
        /** Implementation details and semantic operations. */
        if(lower) {
            text = text.toLowerCase();
        }
        /** Implementation details and semantic operations. */
        if(stripAccents) {
            text = text.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
        }
        return text;
    }

    /** Implementation details and semantic operations. */

    public preTokenize(text: string): string[] {
        const matches = text.match(/\w+|[^\w\s]/gu);
        return matches ? Array.from(matches) : [];
    }
}

/** Implementation details and semantic operations. */

export class BPETokenizer extends Tokenizer {
    public merges: Map<string, number>;
    public cache: Map<string, string[]>;
    public byteFallback: boolean;

    /** Implementation details and semantic operations. */

    constructor(
        vocab: Record<string, number>,
        merges: Record<string, number>,
        byteFallback: boolean = false,
        kwargs: { unkToken?: string; padToken?: string; clsToken?: string; sepToken?: string; maskToken?: string } = {}
    ) {
        /** Implementation details and semantic operations. */
        super(
            vocab,
            kwargs.unkToken,
            kwargs.padToken,
            kwargs.clsToken,
            kwargs.sepToken,
            kwargs.maskToken
        );
        this.merges = new Map(Object.entries(merges));
        this.cache = new Map();
        this.byteFallback = byteFallback;
    }

    /** Implementation details and semantic operations. */

    private getPairs(word: string[]): Set<string> {
        const pairs = new Set<string>();
        /** Implementation details and semantic operations. */
        if(word.length < 2) return pairs;
        let prevChar = word[0];
        /** Implementation details and semantic operations. */
        for(let i = 1; i < word.length; i++) {
            const char = word[i];
            pairs.add(`${prevChar},${char}`);
            prevChar = char;
        }
        return pairs;
    }

    /** Implementation details and semantic operations. */

    public bpe(token: string): string[] {
        /** Implementation details and semantic operations. */
        if(this.cache.has(token)) {
            return this.cache.get(token)!;
        }

        let word = Array.from(token);
        /** Implementation details and semantic operations. */
        if(word.length === 0) return [];

        let pairs = this.getPairs(word);
        /** Implementation details and semantic operations. */
        if(pairs.size === 0) return [token];

        /** Implementation details and semantic operations. */

        while(true) {
            let bigram: string | null = null;
            let minRank = Infinity;

            /** Implementation details and semantic operations. */

            for(const pairStr of pairs) {
                const rank = this.merges.get(pairStr.replace(",", " ")) ?? 1e9;
                /** Implementation details and semantic operations. */
                if(rank < minRank) {
                    minRank = rank;
                    bigram = pairStr;
                }
            }

            /** Implementation details and semantic operations. */

            if(bigram === null || !this.merges.has(bigram.replace(",", " "))) {
                break;
            }

            const [first, second] = bigram.split(",");
            let newWord: string[] = [];
            let i = 0;

            /** Implementation details and semantic operations. */

            while(i < word.length) {
                const j = word.indexOf(first, i);
                /** Implementation details and semantic operations. */
                if(j === -1) {
                    newWord.push(...word.slice(i));
                    break;
                }

                newWord.push(...word.slice(i, j));
                i = j;

                /** Implementation details and semantic operations. */

                if(i < word.length - 1 && word[i] === first && word[i + 1] === second) {
                    newWord.push(first + second);
                    i += 2;
                } else {
                    newWord.push(word[i]);
                    i += 1;
                }
            }

            word = newWord;
            /** Implementation details and semantic operations. */
            if(word.length === 1) break;
            pairs = this.getPairs(word);
        }

        this.cache.set(token, word);
        return word;
    }

    /** Implementation details and semantic operations. */

    public encode(text: string): number[] {
        text = this.normalize(text, false, false);
        const tokens = this.preTokenize(text);
        const ids: number[] = [];

        /** Implementation details and semantic operations. */

        for(const token of tokens) {
            const bpeTokens = this.bpe(token);
            /** Implementation details and semantic operations. */
            for(const bpeToken of bpeTokens) {
                /** Implementation details and semantic operations. */
                if(this.vocab.has(bpeToken)) {
                    ids.push(this.vocab.get(bpeToken)!);
                } else {
                    /** Implementation details and semantic operations. */
                    if(this.byteFallback) {
                        const encoder = new TextEncoder();
                        const bytes = encoder.encode(bpeToken);
                        /** Implementation details and semantic operations. */
                        for(const byte of bytes) {
                            const byteStr = `<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`;
                            ids.push(this.vocab.get(byteStr) ?? this.unkId);
                        }
                    } else {
                        ids.push(this.unkId);
                    }
                }
            }
        }
        return ids;
    }

    /** Implementation details and semantic operations. */

    public decode(ids: number[]): string {
        const textTokens: string[] = [];
        /** Implementation details and semantic operations. */
        for(const id of ids) {
            textTokens.push(this.inverseVocab.get(id) ?? this.unkToken);
        }

        let text = textTokens.join("");
        text = text.replace(/Ġ/g, " ");
        text = text.replace(/ /g, " ");
        return text.trim();
    }

    public static fromHuggingFace(jsonStr: string): BPETokenizer {
        const data = JSON.parse(jsonStr);
        const modelData = data.model || {};
        const vocab = modelData.vocab || {};

        const mergesList: string[] = modelData.merges || [];
        const merges: Record<string, number> = {};
        /** Implementation details and semantic operations. */
        for(let i = 0; i < mergesList.length; i++) {
            merges[mergesList[i]] = i;
        }

        const addedTokens = data.added_tokens || [];
        /** Implementation details and semantic operations. */
        for(const tokenData of addedTokens) {
            const content = tokenData.content;
            const idVal = tokenData.id;
            /** Implementation details and semantic operations. */
            if(content && typeof idVal === "number") {
                vocab[content] = idVal;
            }
        }

        return new BPETokenizer(vocab, merges);
    }
}

/** Implementation details and semantic operations. */

export class WordPieceTokenizer extends Tokenizer {
    public maxInputCharsPerWord: number;

    /** Implementation details and semantic operations. */

    constructor(
        vocab: Record<string, number>,
        maxInputCharsPerWord: number = 100,
        kwargs: { unkToken?: string; padToken?: string; clsToken?: string; sepToken?: string; maskToken?: string } = {}
    ) {
        /** Implementation details and semantic operations. */
        super(
            vocab,
            kwargs.unkToken,
            kwargs.padToken,
            kwargs.clsToken,
            kwargs.sepToken,
            kwargs.maskToken
        );
        this.maxInputCharsPerWord = maxInputCharsPerWord;
    }

    /** Implementation details and semantic operations. */

    public encode(text: string): number[] {
        text = this.normalize(text);
        const tokens = this.preTokenize(text);
        const outputIds: number[] = [];

        /** Implementation details and semantic operations. */

        for(const token of tokens) {
            const chars = Array.from(token);
            /** Implementation details and semantic operations. */
            if(chars.length > this.maxInputCharsPerWord) {
                outputIds.push(this.unkId);
                continue;
            }

            let isBad = false;
            let start = 0;
            const subTokens: number[] = [];

            /** Implementation details and semantic operations. */

            while(start < chars.length) {
                let end = chars.length;
                let curSubstr: string | null = null;
                let curId = -1;

                /** Implementation details and semantic operations. */

                while(start < end) {
                    let substr = chars.slice(start, end).join("");
                    /** Implementation details and semantic operations. */
                    if(start > 0) {
                        substr = "##" + substr;
                    }

                    /** Implementation details and semantic operations. */

                    if(this.vocab.has(substr)) {
                        curSubstr = substr;
                        curId = this.vocab.get(substr)!;
                        break;
                    }
                    end -= 1;
                }

                /** Implementation details and semantic operations. */

                if(curSubstr === null) {
                    isBad = true;
                    break;
                }
                subTokens.push(curId);
                start = end;
            }

            /** Implementation details and semantic operations. */

            if(isBad) {
                outputIds.push(this.unkId);
            } else {
                outputIds.push(...subTokens);
            }
        }
        return outputIds;
    }

    /** Implementation details and semantic operations. */

    public decode(ids: number[]): string {
        const tokens: string[] = [];
        /** Implementation details and semantic operations. */
        for(const id of ids) {
            tokens.push(this.inverseVocab.get(id) ?? this.unkToken);
        }

        let outStr = "";
        /** Implementation details and semantic operations. */
        for(const token of tokens) {
            /** Implementation details and semantic operations. */
            if(token.startsWith("##")) {
                outStr += token.slice(2);
            } else {
                /** Implementation details and semantic operations. */
                if(outStr.length > 0) {
                    outStr += " ";
                }
                outStr += token;
            }
        }
        return outStr;
    }
}