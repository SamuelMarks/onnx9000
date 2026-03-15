/** Implementation details and semantic operations. */
export function rotatedNms(
    boxes: number[][],
    scores: number[],
    iouThreshold: number,
    scoreThreshold: number = 0.0
): number[] {
    const boundedBoxes: number[][] = [];
    /** Implementation details and semantic operations. */
    for(const box of boxes) {
        const [cx, cy, w, h, angle] = box;
        const cosA = Math.abs(Math.cos(angle));
        const sinA = Math.abs(Math.sin(angle));

        const bw = w * cosA + h * sinA;
        const bh = w * sinA + h * cosA;

        boundedBoxes.push([
            cx - bw / 2, cy - bh / 2,
            cx + bw / 2, cy + bh / 2
        ]);
    }

    let indices: number[] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < scores.length; i++) {
        /** Implementation details and semantic operations. */
        if(scores[i] >= scoreThreshold) indices.push(i);
    }
    indices.sort((a, b) => scores[b] - scores[a]);

    const keep: number[] = [];
    /** Implementation details and semantic operations. */
    while(indices.length > 0) {
        const curr = indices.shift()!;
        keep.push(curr);

        const currBox = boundedBoxes[curr];
        indices = indices.filter(i => {
            const box = boundedBoxes[i];
            const x1 = Math.max(currBox[0], box[0]);
            const y1 = Math.max(currBox[1], box[1]);
            const x2 = Math.min(currBox[2], box[2]);
            const y2 = Math.min(currBox[3], box[3]);

            const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
            /** Implementation details and semantic operations. */
            if(interArea === 0) return true;

            const area1 = (currBox[2] - currBox[0]) * (currBox[3] - currBox[1]);
            const area2 = (box[2] - box[0]) * (box[3] - box[1]);
            const iou = interArea / (area1 + area2 - interArea);
            return iou <= iouThreshold;
        });
    }
    return keep;
}

/** Implementation details and semantic operations. */

export function gridSample(
    inputTensor: number[][][][],
    grid: number[][][][],
    mode: string = "bilinear",
    paddingMode: string = "zeros",
    alignCorners: boolean = false
): number[][][][] {
    const N = inputTensor.length;
    const C = N > 0 ? inputTensor[0].length : 0;
    const H = C > 0 ? inputTensor[0][0].length : 0;
    const W = H > 0 ? inputTensor[0][0][0].length : 0;

    const HOut = grid.length > 0 ? grid[0].length : 0;
    const WOut = HOut > 0 ? grid[0][0].length : 0;

    const out: number[][][][] = [];
    /** Implementation details and semantic operations. */
    for(let n = 0; n < N; n++) {
        const batch: number[][][] = [];
        /** Implementation details and semantic operations. */
        for(let c = 0; c < C; c++) {
            const channel: number[][] = [];
            /** Implementation details and semantic operations. */
            for(let h = 0; h < HOut; h++) {
                channel.push([...Array(WOut)].map(() => 0.0));
            }
            batch.push(channel);
        }
        out.push(batch);
    }

    /** Implementation details and semantic operations. */

    for(let n = 0; n < N; n++) {
        /** Implementation details and semantic operations. */
        for(let hOut = 0; hOut < HOut; hOut++) {
            /** Implementation details and semantic operations. */
            for(let wOut = 0; wOut < WOut; wOut++) {
                const [x, y] = grid[n][hOut][wOut];

                let ix: number, iy: number;
                /** Implementation details and semantic operations. */
                if(alignCorners) {
                    ix = ((x + 1) / 2) * (W - 1);
                    iy = ((y + 1) / 2) * (H - 1);
                } else {
                    ix = ((x + 1) * W - 1) / 2;
                    iy = ((y + 1) * H - 1) / 2;
                }

                /** Implementation details and semantic operations. */

                if(paddingMode === "border") {
                    ix = Math.max(0.0, Math.min(W - 1.0, ix));
                    iy = Math.max(0.0, Math.min(H - 1.0, iy));
                } else if (paddingMode === "reflection") {
                    /** Implementation details and semantic operations. */
                    if(ix < 0) ix = -ix;
                    /** Implementation details and semantic operations. */
                    if(ix >= W) ix = W - 1 - (ix - W);
                    /** Implementation details and semantic operations. */
                    if(iy < 0) iy = -iy;
                    /** Implementation details and semantic operations. */
                    if(iy >= H) iy = H - 1 - (iy - H);
                }

                /** Implementation details and semantic operations. */

                if(mode === "nearest") {
                    const ixN = Math.round(ix);
                    const iyN = Math.round(iy);
                    /** Implementation details and semantic operations. */
                    for(let c = 0; c < C; c++) {
                        /** Implementation details and semantic operations. */
                        if(ixN >= 0 && ixN < W && iyN >= 0 && iyN < H) {
                            out[n][c][hOut][wOut] = inputTensor[n][c][iyN][ixN];
                        }
                    }
                } else {
                    const ixNw = Math.floor(ix);
                    const iyNw = Math.floor(iy);
                    const ixNe = ixNw + 1;
                    const iySw = iyNw + 1;

                    const dx = ix - ixNw;
                    const dy = iy - iyNw;

                    /** Implementation details and semantic operations. */

                    for(let c = 0; c < C; c++) {
                        const vNw = (ixNw >= 0 && ixNw < W && iyNw >= 0 && iyNw < H) ? inputTensor[n][c][iyNw][ixNw] : 0.0;
                        const vNe = (ixNe >= 0 && ixNe < W && iyNw >= 0 && iyNw < H) ? inputTensor[n][c][iyNw][ixNe] : 0.0;
                        const vSw = (ixNw >= 0 && ixNw < W && iySw >= 0 && iySw < H) ? inputTensor[n][c][iySw][ixNw] : 0.0;
                        const vSe = (ixNe >= 0 && ixNe < W && iySw >= 0 && iySw < H) ? inputTensor[n][c][iySw][ixNe] : 0.0;

                        out[n][c][hOut][wOut] = (
                            vNw * (1 - dx) * (1 - dy) +
                            vNe * dx * (1 - dy) +
                            vSw * (1 - dx) * dy +
                            vSe * dx * dy
                        );
                    }
                }
            }
        }
    }
    return out;
}

/** Implementation details and semantic operations. */

export function roiAlign(
    inputTensor: number[][][][],
    rois: number[][],
    batchIndices: number[],
    outputHeight: number,
    outputWidth: number,
    spatialScale: number,
    samplingRatio: number,
    aligned: boolean = true
): number[][][][] {
    const numRois = rois.length;
    const C = inputTensor[0].length;
    const H = inputTensor[0][0].length;
    const W = inputTensor[0][0][0].length;

    const out: number[][][][] = [];
    /** Implementation details and semantic operations. */
    for(let rIdx = 0; rIdx < numRois; rIdx++) {
        const roiChannels: number[][][] = [];
        /** Implementation details and semantic operations. */
        for(let c = 0; c < C; c++) {
            const plane: number[][] = [];
            /** Implementation details and semantic operations. */
            for(let ph = 0; ph < outputHeight; ph++) {
                plane.push([...Array(outputWidth)].map(() => 0.0));
            }
            roiChannels.push(plane);
        }
        out.push(roiChannels);
    }

    const offset = aligned ? 0.5 : 0.0;

    /** Implementation details and semantic operations. */

    for(let rIdx = 0; rIdx < numRois; rIdx++) {
        const bIdx = batchIndices[rIdx];
        let [x1, y1, x2, y2] = rois[rIdx];

        x1 = x1 * spatialScale - offset;
        y1 = y1 * spatialScale - offset;
        x2 = x2 * spatialScale - offset;
        y2 = y2 * spatialScale - offset;

        const roiW = Math.max(x2 - x1, 1.0);
        const roiH = Math.max(y2 - y1, 1.0);

        const binSizeH = roiH / outputHeight;
        const binSizeW = roiW / outputWidth;

        const gridH = samplingRatio > 0 ? samplingRatio : Math.ceil(roiH / outputHeight);
        const gridW = samplingRatio > 0 ? samplingRatio : Math.ceil(roiW / outputWidth);

        /** Implementation details and semantic operations. */

        for(let c = 0; c < C; c++) {
            /** Implementation details and semantic operations. */
            for(let ph = 0; ph < outputHeight; ph++) {
                /** Implementation details and semantic operations. */
                for(let pw = 0; pw < outputWidth; pw++) {
                    let val = 0.0;
                    /** Implementation details and semantic operations. */
                    for(let iy = 0; iy < gridH; iy++) {
                        /** Implementation details and semantic operations. */
                        for(let ix = 0; ix < gridW; ix++) {
                            let y = y1 + ph * binSizeH + (iy + 0.5) * binSizeH / gridH;
                            let x = x1 + pw * binSizeW + (ix + 0.5) * binSizeW / gridW;

                            /** Implementation details and semantic operations. */

                            if(y < -1.0 || y > H || x < -1.0 || x > W) continue;

                            y = Math.max(0.0, Math.min(H - 1.0, y));
                            x = Math.max(0.0, Math.min(W - 1.0, x));

                            const yLow = Math.floor(y);
                            const xLow = Math.floor(x);
                            const yHigh = Math.min(yLow + 1, H - 1);
                            const xHigh = Math.min(xLow + 1, W - 1);

                            const dy = y - yLow;
                            const dx = x - xLow;

                            const w1 = (1 - dx) * (1 - dy);
                            const w2 = dx * (1 - dy);
                            const w3 = (1 - dx) * dy;
                            const w4 = dx * dy;

                            val += (
                                w1 * inputTensor[bIdx][c][yLow][xLow] +
                                w2 * inputTensor[bIdx][c][yLow][xHigh] +
                                w3 * inputTensor[bIdx][c][yHigh][xLow] +
                                w4 * inputTensor[bIdx][c][yHigh][xHigh]
                            );
                        }
                    }
                    out[rIdx][c][ph][pw] = val / (gridH * gridW);
                }
            }
        }
    }
    return out;
}

/** Implementation details and semantic operations. */

export function deformConv2d(
    x: number[][][][],
    weight: number[][][][],
    offset: number[][][][],
    mask: number[][][][] | null = null,
    bias: number[] | null = null,
    stride: number = 1,
    padding: number = 0,
    dilation: number = 1
): number[][][][] {
    const N = x.length;
    const inC = x[0].length;
    const inH = x[0][0].length;
    const inW = x[0][0][0].length;

    const outC = weight.length;
    const kH = weight[0][0].length;
    const kW = weight[0][0][0].length;

    const outH = Math.floor((inH + 2 * padding - dilation * (kH - 1) - 1) / stride) + 1;
    const outW = Math.floor((inW + 2 * padding - dilation * (kW - 1) - 1) / stride) + 1;

    const out: number[][][][] = [];
    /** Implementation details and semantic operations. */
    for(let n = 0; n < N; n++) {
        const batch: number[][][] = [];
        /** Implementation details and semantic operations. */
        for(let oc = 0; oc < outC; oc++) {
            const channel: number[][] = [];
            /** Implementation details and semantic operations. */
            for(let oh = 0; oh < outH; oh++) {
                channel.push([...Array(outW)].map(() => 0.0));
            }
            batch.push(channel);
        }
        out.push(batch);
    }

    /** Implementation details and semantic operations. */

    for(let n = 0; n < N; n++) {
        /** Implementation details and semantic operations. */
        for(let oc = 0; oc < outC; oc++) {
            /** Implementation details and semantic operations. */
            for(let oh = 0; oh < outH; oh++) {
                /** Implementation details and semantic operations. */
                for(let ow = 0; outW > ow; ow++) {
                    let val = bias ? bias[oc] : 0.0;
                    /** Implementation details and semantic operations. */
                    for(let ic = 0; ic < inC; ic++) {
                        /** Implementation details and semantic operations. */
                        for(let kh = 0; kh < kH; kh++) {
                            /** Implementation details and semantic operations. */
                            for(let kw = 0; kw < kW; kw++) {
                                const ih = oh * stride - padding + kh * dilation;
                                const iw = ow * stride - padding + kw * dilation;

                                const offsetIdxY = 2 * (kh * kW + kw);
                                const offsetIdxX = offsetIdxY + 1;

                                const dy = offset[n][offsetIdxY][oh][ow];
                                const dx = offset[n][offsetIdxX][oh][ow];

                                const sampledY = ih + dy;
                                const sampledX = iw + dx;

                                let mVal = 1.0;
                                /** Implementation details and semantic operations. */
                                if(mask) {
                                    const maskIdx = kh * kW + kw;
                                    mVal = mask[n][maskIdx][oh][ow];
                                }

                                /** Implementation details and semantic operations. */

                                if(sampledY >= 0 && sampledY < inH && sampledX >= 0 && sampledX < inW) {
                                    const y0 = Math.floor(sampledY);
                                    const x0 = Math.floor(sampledX);
                                    const y1 = Math.min(y0 + 1, inH - 1);
                                    const x1 = Math.min(x0 + 1, inW - 1);

                                    const wy = sampledY - y0;
                                    const wx = sampledX - x0;

                                    const v00 = x[n][ic][y0][x0];
                                    const v01 = x[n][ic][y0][x1];
                                    const v10 = x[n][ic][y1][x0];
                                    const v11 = x[n][ic][y1][x1];

                                    const sVal = (
                                        v00 * (1 - wy) * (1 - wx) +
                                        v01 * (1 - wy) * wx +
                                        v10 * wy * (1 - wx) +
                                        v11 * wy * wx
                                    );

                                    val += sVal * weight[oc][ic][kh][kw] * mVal;
                                }
                            }
                        }
                    }
                    out[n][oc][oh][ow] = val;
                }
            }
        }
    }
    return out;
}
