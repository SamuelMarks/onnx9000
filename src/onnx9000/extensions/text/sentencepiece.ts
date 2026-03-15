import { UnigramTokenizer } from "./unigram";
import { parseSPMModel } from "./spm";

/** Implementation details and semantic operations. */

export function spmNormalize(text: string): string {
    return text.normalize("NFKC");
}

/** Implementation details and semantic operations. */

export function spmPreTokenize(text: string): string {
    return text.replace(/ /g, " ");
}

/** Implementation details and semantic operations. */

export function spmDecode(tokens: string[]): string {
    let text = tokens.join("");
    text = text.replace(/ /g, " ");
    return text;
}

/** Implementation details and semantic operations. */

export class SentencePieceTokenizer extends UnigramTokenizer {
    public byteFallback: boolean;

    /** Implementation details and semantic operations. */

    constructor(
        vocab: Record<string, number>,
        scores: Record<string, number>,
        byteFallback: boolean = false,
        kwargs: { unkToken?: string; padToken?: string; clsToken?: string; sepToken?: string; maskToken?: string } = {}
    ) {
        /** Implementation details and semantic operations. */
        super(vocab, scores, -100.0, kwargs);
        this.byteFallback = byteFallback;
    }

    /** Implementation details and semantic operations. */

    public normalize(text: string, lower: boolean = false, stripAccents: boolean = false): string {
        text = super.normalize(text, lower, stripAccents);
        return spmNormalize(text);
    }

    /** Implementation details and semantic operations. */

    public encode(text: string): number[] {
        text = this.normalize(text);
        text = spmPreTokenize(text);
        return super.encode(text);
    }

    /** Implementation details and semantic operations. */

    public decode(ids: number[]): string {
        const tokens: string[] = [];
        /** Implementation details and semantic operations. */
        for(const id of ids) {
            tokens.push(this.inverseVocab.get(id) ?? this.unkToken);
        }

        const decodedTokens: string[] = [];
        let byteBuffer: number[] = [];

        /** Implementation details and semantic operations. */

        for(const t of tokens) {
            /** Implementation details and semantic operations. */
            if(t.startsWith("<0x") && t.endsWith(">") && t.length === 6) {
                const hexStr = t.slice(3, 5);
                const b = parseInt(hexStr, 16);
                /** Implementation details and semantic operations. */
                if(!isNaN(b)) {
                    byteBuffer.push(b);
                    continue;
                }
            }

            /** Implementation details and semantic operations. */

            if(byteBuffer.length > 0) {
                const decoder = new TextDecoder("utf-8", { fatal: false });
                decodedTokens.push(decoder.decode(new Uint8Array(byteBuffer)));
                byteBuffer = [];
            }
            decodedTokens.push(t);
        }

        /** Implementation details and semantic operations. */

        if(byteBuffer.length > 0) {
            const decoder = new TextDecoder("utf-8", { fatal: false });
            decodedTokens.push(decoder.decode(new Uint8Array(byteBuffer)));
        }

        return spmDecode(decodedTokens);
    }

    public static fromSPMBuffer(buffer: Uint8Array, byteFallback: boolean = false): SentencePieceTokenizer {
        const pieces = parseSPMModel(buffer);
        const vocab: Record<string, number> = {};
        const scores: Record<string, number> = {};

        /** Implementation details and semantic operations. */

        for(let i = 0; i < pieces.length; i++) {
            const p = pieces[i];
            vocab[p.piece] = i;
            scores[p.piece] = p.score;
        }

        return new SentencePieceTokenizer(vocab, scores, byteFallback);
    }
}
