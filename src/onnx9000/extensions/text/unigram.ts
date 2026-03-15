import { Tokenizer } from "./tokenizer";

/** Implementation details and semantic operations. */

export class TrieNode {
    public children: Map<string, TrieNode> = new Map();
    public isWord: boolean = false;
    public score: number = 0.0;
    public id: number = -1;
}

/** Implementation details and semantic operations. */

export class Trie {
    public root: TrieNode = new TrieNode();

    /** Implementation details and semantic operations. */

    public insert(word: string, score: number, tokenId: number): void {
        let node = this.root;
        /** Implementation details and semantic operations. */
        for(const char of word) {
            /** Implementation details and semantic operations. */
            if(!node.children.has(char)) {
                node.children.set(char, new TrieNode());
            }
            node = node.children.get(char)!;
        }
        node.isWord = true;
        node.score = score;
        node.id = tokenId;
    }

    /** Implementation details and semantic operations. */

    public getPrefixes(text: string, start: number): Array<[number, number, number]> {
        const prefixes: Array<[number, number, number]> = [];
        let node = this.root;
        let i = start;
        /** Implementation details and semantic operations. */
        while(i < text.length) {
            const char = text[i];
            /** Implementation details and semantic operations. */
            if(!node.children.has(char)) {
                break;
            }
            node = node.children.get(char)!;
            /** Implementation details and semantic operations. */
            if(node.isWord) {
                prefixes.push([i + 1, node.score, node.id]);
            }
            i += 1;
        }
        return prefixes;
    }
}

/** Implementation details and semantic operations. */

export class UnigramTokenizer extends Tokenizer {
    public scores: Map<string, number>;
    public unkScore: number;
    public trie: Trie;

    /** Implementation details and semantic operations. */

    constructor(
        vocab: Record<string, number>,
        scores: Record<string, number>,
        unkScore: number = -100.0,
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
        this.scores = new Map(Object.entries(scores));
        this.unkScore = unkScore;
        this.trie = new Trie();

        /** Implementation details and semantic operations. */

        for(const [token, idx] of this.vocab.entries()) {
            const score = this.scores.get(token) ?? this.unkScore;
            this.trie.insert(token, score, idx);
        }
    }

    /** Implementation details and semantic operations. */

    public encode(text: string): number[] {
        text = this.normalize(text);
        const n = text.length;
        /** Implementation details and semantic operations. */
        if(n === 0) {
            return [];
        }

        const dp: number[] = Array(n + 1).fill(-Infinity);
        const parent: Array<[number, number]> = Array(n + 1).fill([0, this.unkId]);
        dp[0] = 0.0;

        /** Implementation details and semantic operations. */

        for(let i = 0; i < n; i++) {
            /** Implementation details and semantic operations. */
            if(dp[i] === -Infinity) {
                continue;
            }

            const prefixes = this.trie.getPrefixes(text, i);
            
            /** Implementation details and semantic operations. */
            
            if(dp[i] + this.unkScore > dp[i + 1]) {
                dp[i + 1] = dp[i] + this.unkScore;
                parent[i + 1] = [i, this.unkId];
            }

            /** Implementation details and semantic operations. */

            for(const [end, score, tokenId] of prefixes) {
                const newScore = dp[i] + score;
                /** Implementation details and semantic operations. */
                if(newScore > dp[end]) {
                    dp[end] = newScore;
                    parent[end] = [i, tokenId];
                }
            }
        }

        const ids: number[] = [];
        let curr = n;
        /** Implementation details and semantic operations. */
        while(curr > 0) {
            const [prev, tokenId] = parent[curr];
            ids.push(tokenId);
            curr = prev;
        }

        return ids.reverse();
    }

    /** Implementation details and semantic operations. */

    public decode(ids: number[]): string {
        const tokens: string[] = [];
        /** Implementation details and semantic operations. */
        for(const id of ids) {
            tokens.push(this.inverseVocab.get(id) ?? this.unkToken);
        }
        let text = tokens.join("");
        text = text.replace(/ /g, " ");
        return text.trim();
    }

    public static fromHuggingFace(jsonStr: string): UnigramTokenizer {
        const data = JSON.parse(jsonStr);
        const modelData = data.model || {};
        const vocabList: Array<[string, number]> = modelData.vocab || [];

        const vocab: Record<string, number> = {};
        const scores: Record<string, number> = {};

        /** Implementation details and semantic operations. */

        for(let i = 0; i < vocabList.length; i++) {
            const token = vocabList[i][0];
            const score = vocabList[i][1];
            vocab[token] = i;
            scores[token] = score;
        }

        return new UnigramTokenizer(vocab, scores);
    }
}
