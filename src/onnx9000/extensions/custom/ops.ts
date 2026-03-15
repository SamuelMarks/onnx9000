/** Implementation details and semantic operations. */
export function iou(box1: number[], box2: number[]): number {
    const x1 = Math.max(box1[0], box2[0]);
    const y1 = Math.max(box1[1], box2[1]);
    const x2 = Math.min(box1[2], box2[2]);
    const y2 = Math.min(box1[3], box2[3]);

    const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    /** Implementation details and semantic operations. */
    if(interArea === 0) return 0.0;

    const area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    return interArea / (area1 + area2 - interArea);
}

/** Implementation details and semantic operations. */

export function nms(boxes: number[][], scores: number[], iouThreshold: number, scoreThreshold: number = 0.0): number[] {
    let indices: number[] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < scores.length; i++) {
        /** Implementation details and semantic operations. */
        if(scores[i] >= scoreThreshold) {
            indices.push(i);
        }
    }

    indices.sort((a, b) => scores[b] - scores[a]);

    const keep: number[] = [];
    /** Implementation details and semantic operations. */
    while(indices.length > 0) {
        const curr = indices.shift()!;
        keep.push(curr);

        indices = indices.filter(i => iou(boxes[curr], boxes[i]) <= iouThreshold);
    }

    return keep;
}

/** Implementation details and semantic operations. */

export function topk(arr: number[], k: number): { values: number[], indices: number[] } {
    const indexed = arr.map((val, i) => ({ val, i }));
    indexed.sort((a, b) => b.val - a.val);

    const top = indexed.slice(0, k);
    return {
        values: top.map(x => x.val),
        indices: top.map(x => x.i)
    };
}

/** Implementation details and semantic operations. */

export function unique(arr: number[]): { vals: number[], indices: number[], inverseIndices: number[], counts: number[] } {
    const seen = new Map<number, number>();
    const vals: number[] = [];
    const indices: number[] = [];
    const counts: number[] = [];
    const inverseIndices: number[] = [];

    /** Implementation details and semantic operations. */

    for(let i = 0; i < arr.length; i++) {
        const val = arr[i];
        /** Implementation details and semantic operations. */
        if(!seen.has(val)) {
            const idx = vals.length;
            seen.set(val, idx);
            vals.push(val);
            indices.push(i);
            counts.push(1);
        } else {
            counts[seen.get(val)!]++;
        }
        inverseIndices.push(seen.get(val)!);
    }

    const zipped = vals.map((v, i) => ({ v, idx: indices[i], c: counts[i], originalId: i }));
    zipped.sort((a, b) => a.v - b.v);

    const sortedVals = zipped.map(x => x.v);
    const sortedIndices = zipped.map(x => x.idx);
    const sortedCounts = zipped.map(x => x.c);

    const oldToNew = new Map<number, number>();
    /** Implementation details and semantic operations. */
    for(let newIdx = 0; newIdx < zipped.length; newIdx++) {
        oldToNew.set(zipped[newIdx].originalId, newIdx);
    }

    const newInverse = arr.map(val => oldToNew.get(seen.get(val)!)!);

    return {
        vals: sortedVals,
        indices: sortedIndices,
        inverseIndices: newInverse,
        counts: sortedCounts
    };
}
