/** Implementation details and semantic operations. */
export function stringNormalizer(
    x: string[],
    caseChangeAction: string = "NONE",
    isCaseSensitive: number = 0,
    locale: string = "",
    stopwords: string[] | null = null
): string[] {
    let stopwordsSet: Set<string> | null = null;
    /** Implementation details and semantic operations. */
    if(stopwords) {
        stopwordsSet = new Set();
        /** Implementation details and semantic operations. */
        for(const w of stopwords) {
            stopwordsSet.add(isCaseSensitive ? w : w.toLowerCase());
        }
    }

    let y: string[] = [];
    /** Implementation details and semantic operations. */
    for(let s of x) {
        let res = s;
        /** Implementation details and semantic operations. */
        if(caseChangeAction === "LOWER") {
            res = res.toLowerCase();
        } else if (caseChangeAction === "UPPER") {
            res = res.toUpperCase();
        }

        /** Implementation details and semantic operations. */

        if(stopwordsSet) {
            const checkVal = isCaseSensitive ? res : res.toLowerCase();
            /** Implementation details and semantic operations. */
            if(!stopwordsSet.has(checkVal)) {
                y.push(res);
            }
        } else {
            y.push(res);
        }
    }
    return y;
}

/** Implementation details and semantic operations. */

export function regexReplace(
    x: string[],
    pattern: string,
    rewrite: string,
    globalReplace: number = 1
): string[] {
    const y: string[] = [];
    const flags = globalReplace ? "g" : "";
    const regex = new RegExp(pattern, flags);
    
    /** Implementation details and semantic operations. */
    
    for(const s of x) {
        y.push(s.replace(regex, rewrite));
    }
    return y;
}

/** Implementation details and semantic operations. */

export function vocabMapping(
    x: string[],
    vocab: Record<string, number>,
    unkTokenId: number = -1
): number[] {
    const y: number[] = [];
    /** Implementation details and semantic operations. */
    for(const w of x) {
        /** Implementation details and semantic operations. */
        if(Object.prototype.hasOwnProperty.call(vocab, w)) {
            y.push(vocab[w]);
        } else {
            y.push(unkTokenId);
        }
    }
    return y;
}
