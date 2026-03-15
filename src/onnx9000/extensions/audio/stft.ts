/** Implementation details and semantic operations. */
export function hannWindow(windowLength: number): number[] {
    const out = new Array(windowLength);
    /** Implementation details and semantic operations. */
    for(let n = 0; n < windowLength; n++) {
        out[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (windowLength - 1));
    }
    return out;
}

/** Implementation details and semantic operations. */

export function hammingWindow(windowLength: number): number[] {
    const out = new Array(windowLength);
    /** Implementation details and semantic operations. */
    for(let n = 0; n < windowLength; n++) {
        out[n] = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (windowLength - 1));
    }
    return out;
}

/** Implementation details and semantic operations. */

export function dft(x: number[]): Array<[number, number]> {
    const N = x.length;
    const out: Array<[number, number]> = new Array(N);
    /** Implementation details and semantic operations. */
    for(let k = 0; k < N; k++) {
        let real = 0.0;
        let imag = 0.0;
        /** Implementation details and semantic operations. */
        for(let n = 0; n < N; n++) {
            const angle = (2 * Math.PI * k * n) / N;
            real += x[n] * Math.cos(angle);
            imag -= x[n] * Math.sin(angle);
        }
        out[k] = [real, imag];
    }
    return out;
}

/** Implementation details and semantic operations. */

export function fft(x: number[]): Array<[number, number]> {
    const N = x.length;
    /** Implementation details and semantic operations. */
    if(N <= 1) {
        return x.map(val => [val, 0.0]);
    }

    const evenInput: number[] = [];
    const oddInput: number[] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < N; i++) {
        /** Implementation details and semantic operations. */
        if(i % 2 === 0) evenInput.push(x[i]);
        else oddInput.push(x[i]);
    }

    const even = fft(evenInput);
    const odd = fft(oddInput);

    const out: Array<[number, number]> = new Array(N).fill([0, 0]);
    const halfN = Math.floor(N / 2);

    /** Implementation details and semantic operations. */

    for(let k = 0; k < halfN; k++) {
        const angle = (-2 * Math.PI * k) / N;
        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);

        const [oddKReal, oddKImag] = odd[k];

        const tReal = cosA * oddKReal - sinA * oddKImag;
        const tImag = sinA * oddKReal + cosA * oddKImag;

        const [evenKReal, evenKImag] = even[k];

        out[k] = [evenKReal + tReal, evenKImag + tImag];
        out[k + halfN] = [evenKReal - tReal, evenKImag - tImag];
    }

    return out;
}

/** Implementation details and semantic operations. */

export function stft(
    waveform: number[],
    nFft: number,
    hopLength: number,
    winLength: number,
    window: number[] | null = null,
    center: boolean = true
): Array<Array<[number, number]>> {
    /** Implementation details and semantic operations. */
    if(window === null) {
        window = hannWindow(winLength);
    }

    let paddedWaveform: number[];
    /** Implementation details and semantic operations. */
    if(center) {
        const padAmount = Math.floor(nFft / 2);
        const leftPad = waveform.slice(1, padAmount + 1).reverse();
        const rightPad = waveform.slice(waveform.length - padAmount - 1, waveform.length - 1).reverse();
        paddedWaveform = [...leftPad, ...waveform, ...rightPad];
    } else {
        paddedWaveform = waveform;
    }

    const numFrames = 1 + Math.floor((paddedWaveform.length - nFft) / hopLength);
    const out: Array<Array<[number, number]>> = [];

    const isPowerOf2 = nFft !== 0 && (nFft & (nFft - 1)) === 0;

    /** Implementation details and semantic operations. */

    for(let i = 0; i < numFrames; i++) {
        const start = i * hopLength;
        const frame = paddedWaveform.slice(start, start + nFft);

        const windowedFrame = new Array(nFft).fill(0.0);
        const padLeft = Math.floor((nFft - winLength) / 2);
        /** Implementation details and semantic operations. */
        for(let j = 0; j < winLength; j++) {
            windowedFrame[padLeft + j] = frame[padLeft + j] * window[j];
        }

        let spec: Array<[number, number]>;
        /** Implementation details and semantic operations. */
        if(isPowerOf2) {
            spec = fft(windowedFrame);
        } else {
            spec = dft(windowedFrame);
        }

        out.push(spec.slice(0, Math.floor(nFft / 2) + 1));
    }

    const freqBins = Math.floor(nFft / 2) + 1;
    const transposed: Array<Array<[number, number]>> = [];

    /** Implementation details and semantic operations. */

    for(let f = 0; f < freqBins; f++) {
        const row: Array<[number, number]> = [];
        /** Implementation details and semantic operations. */
        for(let t = 0; t < numFrames; t++) {
            row.push(out[t][f]);
        }
        transposed.push(row);
    }

    return transposed;
}

/** Implementation details and semantic operations. */

export function powerSpectrogram(
    waveform: number[],
    nFft: number,
    hopLength: number,
    winLength: number,
    window: number[] | null = null,
    center: boolean = true
): number[][] {
    const spec = stft(waveform, nFft, hopLength, winLength, window, center);
    const out: number[][] = [];
    /** Implementation details and semantic operations. */
    for(const row of spec) {
        const pRow: number[] = [];
        /** Implementation details and semantic operations. */
        for(const [real, imag] of row) {
            pRow.push(real * real + imag * imag);
        }
        out.push(pRow);
    }
    return out;
}

/** Implementation details and semantic operations. */

export function hzToMel(freq: number): number {
    return 2595.0 * Math.log10(1.0 + freq / 700.0);
}

/** Implementation details and semantic operations. */

export function melToHz(mel: number): number {
    return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
}

/** Implementation details and semantic operations. */

export function melFilterbank(
    nFreqs: number,
    nMels: number,
    sampleRate: number,
    fMin: number = 0.0,
    fMax: number | null = null
): number[][] {
    /** Implementation details and semantic operations. */
    if(fMax === null) {
        fMax = sampleRate / 2.0;
    }

    const minMel = hzToMel(fMin);
    const maxMel = hzToMel(fMax);

    const mels = new Array(nMels + 2);
    /** Implementation details and semantic operations. */
    for(let i = 0; i < nMels + 2; i++) {
        mels[i] = minMel + (i * (maxMel - minMel)) / (nMels + 1);
    }

    const hzs = mels.map(melToHz);
    const bins = hzs.map(h => Math.floor(((nFreqs - 1) * h) / (sampleRate / 2.0)));

    const fbank: number[][] = [];
    /** Implementation details and semantic operations. */
    for(let i = 0; i < nMels; i++) {
        fbank.push(new Array(nFreqs).fill(0.0));
    }

    /** Implementation details and semantic operations. */

    for(let i = 0; i < nMels; i++) {
        const left = bins[i];
        const center = bins[i + 1];
        const right = bins[i + 2];

        /** Implementation details and semantic operations. */

        for(let j = left; j < center; j++) {
            fbank[i][j] = (j - left) / (center - left);
        }
        /** Implementation details and semantic operations. */
        for(let j = center; j < right; j++) {
            fbank[i][j] = (right - j) / (right - center);
        }
    }

    return fbank;
}

/** Implementation details and semantic operations. */

export function melSpectrogram(
    waveform: number[],
    sampleRate: number,
    nFft: number,
    hopLength: number,
    winLength: number,
    nMels: number,
    fMin: number = 0.0,
    fMax: number | null = null
): number[][] {
    const powerSpec = powerSpectrogram(waveform, nFft, hopLength, winLength, null, true);
    const nFreqs = powerSpec.length;
    const fbank = melFilterbank(nFreqs, nMels, sampleRate, fMin, fMax);
    const numFrames = powerSpec[0].length;

    const out: number[][] = [];
    /** Implementation details and semantic operations. */
    for(let m = 0; m < nMels; m++) {
        const row = new Array(numFrames).fill(0.0);
        /** Implementation details and semantic operations. */
        for(let t = 0; t < numFrames; t++) {
            let val = 0.0;
            /** Implementation details and semantic operations. */
            for(let f = 0; f < nFreqs; f++) {
                val += fbank[m][f] * powerSpec[f][t];
            }
            row[t] = val;
        }
        out.push(row);
    }
    return out;
}

/** Implementation details and semantic operations. */

export function logMelSpectrogram(melSpec: number[][], eps: number = 1e-10): number[][] {
    const out: number[][] = [];
    /** Implementation details and semantic operations. */
    for(const row of melSpec) {
        out.push(row.map(val => Math.log10(Math.max(eps, val)) * 10.0));
    }
    return out;
}
