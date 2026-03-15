/** Implementation details and semantic operations. */
export function inverse(matrix: number[][]): number[][] {
    const n = matrix.length;
    /** Implementation details and semantic operations. */
    if(n === 0 || matrix[0].length !== n) {
        throw new Error("Matrix must be square and non-empty");
    }

    const aug: number[][] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < n; i++) {
        const row = [...matrix[i]];
        /** Implementation details and semantic operations. */
        for(let j = 0; j < n; j++) {
            row.push(i === j ? 1.0 : 0.0);
        }
        aug.push(row);
    }

    /** Implementation details and semantic operations. */

    for(let i = 0; i < n; i++) {
        let pivot = aug[i][i];
        /** Implementation details and semantic operations. */
        if(Math.abs(pivot) < 1e-9) {
            let found = false;
            /** Implementation details and semantic operations. */
            for(let j = i + 1; j < n; j++) {
                /** Implementation details and semantic operations. */
                if(Math.abs(aug[j][i]) > 1e-9) {
                    const temp = aug[i];
                    aug[i] = aug[j];
                    aug[j] = temp;
                    pivot = aug[i][i];
                    found = true;
                    break;
                }
            }
            /** Implementation details and semantic operations. */
            if(!found) throw new Error("Matrix is singular");
        }

        /** Implementation details and semantic operations. */

        for(let j = 0; j < 2 * n; j++) {
            aug[i][j] /= pivot;
        }

        /** Implementation details and semantic operations. */

        for(let j = 0; j < n; j++) {
            /** Implementation details and semantic operations. */
            if(i !== j) {
                const factor = aug[j][i];
                /** Implementation details and semantic operations. */
                for(let k = 0; k < 2 * n; k++) {
                    aug[j][k] -= factor * aug[i][k];
                }
            }
        }
    }

    return aug.map(row => row.slice(n));
}

/** Implementation details and semantic operations. */

export function transpose(mat: number[][]): number[][] {
    const out: number[][] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < mat[0].length; i++) {
        const row: number[] = [];
        /** Implementation details and semantic operations. */
        for(let j = 0; j < mat.length; j++) {
            row.push(mat[j][i]);
        }
        out.push(row);
    }
    return out;
}

/** Implementation details and semantic operations. */

export function svd(matrix: number[][], iters: number = 30): { U: number[][], S: number[], V: number[][] } {
    const m = matrix.length;
    const n = matrix[0].length;

    const V: number[][] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < n; i++) {
        const row: number[] = [];
        /** Implementation details and semantic operations. */
        for(let j = 0; j < n; j++) row.push(i === j ? 1.0 : 0.0);
        V.push(row);
    }

    const A = matrix.map(row => [...row]);

    /** Implementation details and semantic operations. */

    for(let it = 0; it < iters; it++) {
        /** Implementation details and semantic operations. */
        for(let p = 0; p < n - 1; p++) {
            /** Implementation details and semantic operations. */
            for(let q = p + 1; q < n; q++) {
                let app = 0, aqq = 0, apq = 0;
                /** Implementation details and semantic operations. */
                for(let i = 0; i < m; i++) {
                    app += A[i][p] * A[i][p];
                    aqq += A[i][q] * A[i][q];
                    apq += A[i][p] * A[i][q];
                }

                /** Implementation details and semantic operations. */

                if(Math.abs(apq) > 1e-9) {
                    const tau = (aqq - app) / (2.0 * apq);
                    let t = 1.0 / (Math.abs(tau) + Math.sqrt(1.0 + tau * tau));
                    /** Implementation details and semantic operations. */
                    if(tau < 0) t = -t;
                    const c = 1.0 / Math.sqrt(1.0 + t * t);
                    const s = t * c;

                    /** Implementation details and semantic operations. */

                    for(let i = 0; i < m; i++) {
                        const aIp = A[i][p];
                        const aIq = A[i][q];
                        A[i][p] = c * aIp - s * aIq;
                        A[i][q] = s * aIp + c * aIq;
                    }

                    /** Implementation details and semantic operations. */

                    for(let i = 0; i < n; i++) {
                        const vIp = V[i][p];
                        const vIq = V[i][q];
                        V[i][p] = c * vIp - s * vIq;
                        V[i][q] = s * vIp + c * vIq;
                    }
                }
            }
        }
    }

    const S: number[] = [];
    /** Implementation details and semantic operations. */
    for(let j = 0; j < n; j++) {
        let sum = 0;
        /** Implementation details and semantic operations. */
        for(let i = 0; i < m; i++) sum += A[i][j] * A[i][j];
        S.push(Math.sqrt(sum));
    }

    const U: number[][] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < m; i++) U.push([...Array(n)].map(() => 0.0));

    /** Implementation details and semantic operations. */

    for(let j = 0; j < n; j++) {
        /** Implementation details and semantic operations. */
        if(S[j] > 1e-9) {
            /** Implementation details and semantic operations. */
            for(let i = 0; i < m; i++) U[i][j] = A[i][j] / S[j];
        } else {
            U[0][j] = 1.0;
        }
    }

    const indexedS = S.map((val, idx) => ({ val, idx }));
    indexedS.sort((a, b) => b.val - a.val);

    const sortedS = indexedS.map(x => x.val);
    const sortedU: number[][] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < m; i++) {
        sortedU.push(indexedS.map(x => U[i][x.idx]));
    }
    const sortedV: number[][] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < n; i++) {
        sortedV.push(indexedS.map(x => V[i][x.idx]));
    }

    return { U: sortedU, S: sortedS, V: transpose(sortedV) };
}
