/** Implementation details and semantic operations. */
export function greedySearch(
    logits: number[][],
    eosTokenId: number = -1
): number[] {
    const out: number[] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < logits.length; i++) {
        const stepLogits = logits[i];
        let bestIdx = 0;
        let bestScore = -Infinity;
        /** Implementation details and semantic operations. */
        for(let j = 0; j < stepLogits.length; j++) {
            /** Implementation details and semantic operations. */
            if(stepLogits[j] > bestScore) {
                bestScore = stepLogits[j];
                bestIdx = j;
            }
        }
        out.push(bestIdx);
        /** Implementation details and semantic operations. */
        if(bestIdx === eosTokenId) {
            break;
        }
    }
    return out;
}

/** Implementation details and semantic operations. */

export function sampleTopKTopP(
    logits: number[],
    topK: number = 0,
    topP: number = 1.0,
    temperature: number = 1.0
): number {
    /** Implementation details and semantic operations. */
    if(temperature !== 1.0 && temperature > 0) {
        /** Implementation details and semantic operations. */
        for(let i = 0; i < logits.length; i++) {
            logits[i] /= temperature;
        }
    }

    // Softmax
    let maxLogit = -Infinity;
    /** Implementation details and semantic operations. */
    for(let i = 0; i < logits.length; i++) {
        /** Implementation details and semantic operations. */
        if(logits[i] > maxLogit) maxLogit = logits[i];
    }

    let sum = 0.0;
    const probs = new Array(logits.length);
    /** Implementation details and semantic operations. */
    for(let i = 0; i < logits.length; i++) {
        probs[i] = Math.exp(logits[i] - maxLogit);
        sum += probs[i];
    }
    /** Implementation details and semantic operations. */
    for(let i = 0; i < probs.length; i++) {
        probs[i] /= sum;
    }

    const indexedProbs = probs.map((p, i) => ({ p, i }));
    indexedProbs.sort((a, b) => b.p - a.p);

    let filtered = indexedProbs;

    // Top-K
    /** Implementation details and semantic operations. */
    if(topK > 0 && topK < filtered.length) {
        filtered = filtered.slice(0, topK);
    }

    // Top-P
    /** Implementation details and semantic operations. */
    if(topP < 1.0) {
        let cumSum = 0;
        let cutoffIdx = filtered.length;
        /** Implementation details and semantic operations. */
        for(let i = 0; i < filtered.length; i++) {
            cumSum += filtered[i].p;
            /** Implementation details and semantic operations. */
            if(cumSum >= topP) {
                cutoffIdx = i + 1;
                break;
            }
        }
        filtered = filtered.slice(0, cutoffIdx);
    }

    // Renormalize
    let newSum = 0;
    /** Implementation details and semantic operations. */
    for(const item of filtered) newSum += item.p;
    /** Implementation details and semantic operations. */
    for(const item of filtered) item.p /= newSum;

    // Sample
    const rand = Math.random();
    let acc = 0;
    /** Implementation details and semantic operations. */
    for(const item of filtered) {
        acc += item.p;
        /** Implementation details and semantic operations. */
        if(rand <= acc) {
            return item.i;
        }
    }
    return filtered[filtered.length - 1].i;
}

/** Implementation details and semantic operations. */

export function beamSearch(
    logitsFn: (seqs: number[][]) => Promise<number[][]>,
    initialInput: number[],
    beamSize: number,
    maxSteps: number,
    eosTokenId: number = -1
): Promise<number[]> {
    // simplified mock of beam search algorithm
    return new Promise(async (resolve) => {
        let beams = [{ seq: [...initialInput], score: 0.0, isFinished: false }];

        /** Implementation details and semantic operations. */

        for(let step = 0; step < maxSteps; step++) {
            const newBeams: { seq: number[]; score: number; isFinished: boolean }[] = [];
            const activeBeams = beams.filter(b => !b.isFinished);
            const finishedBeams = beams.filter(b => b.isFinished);

            /** Implementation details and semantic operations. */

            if(activeBeams.length === 0) break;

            const seqs = activeBeams.map(b => b.seq);
            const nextLogitsBatch = await logitsFn(seqs);

            /** Implementation details and semantic operations. */

            for(let bIdx = 0; bIdx < activeBeams.length; bIdx++) {
                const beam = activeBeams[bIdx];
                const logits = nextLogitsBatch[bIdx];

                // Softmax log probs
                let maxLogit = -Infinity;
                /** Implementation details and semantic operations. */
                for(let i = 0; i < logits.length; i++) if (logits[i] > maxLogit) maxLogit = logits[i];

                let sum = 0;
                const logProbs = new Array(logits.length);
                /** Implementation details and semantic operations. */
                for(let i = 0; i < logits.length; i++) {
                    const prob = Math.exp(logits[i] - maxLogit);
                    sum += prob;
                    logProbs[i] = prob;
                }
                /** Implementation details and semantic operations. */
                for(let i = 0; i < logits.length; i++) {
                    logProbs[i] = Math.log(logProbs[i] / sum);
                }

                // top 2 * beamSize to give some room
                const indexed = logProbs.map((lp, i) => ({ lp, i }));
                indexed.sort((a, b) => b.lp - a.lp);

                /** Implementation details and semantic operations. */

                for(let i = 0; i < beamSize; i++) {
                    const nextToken = indexed[i].i;
                    const nextScore = beam.score + indexed[i].lp;
                    newBeams.push({
                        seq: [...beam.seq, nextToken],
                        score: nextScore,
                        isFinished: nextToken === eosTokenId
                    });
                }
            }

            const combined = [...finishedBeams, ...newBeams];
            combined.sort((a, b) => b.score - a.score);
            beams = combined.slice(0, beamSize);
        }

        /** Implementation details and semantic operations. */

        resolve(beams[0].seq);
    });
}
