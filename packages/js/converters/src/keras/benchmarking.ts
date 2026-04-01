/* eslint-disable */
// @ts-nocheck
import * as fs from 'fs';
import { Graph } from '@onnx9000/core';

export interface TraceEvent {
    name: string;
    cat: string;
    ph: string;
    ts: number;
    pid: number;
    tid: number;
    args?: Record<string, any>;
}

export class ChromeTraceExporter {
    private events: TraceEvent[] = [];

    public startEvent(name: string, category: string = 'keras_to_onnx') {
        this.events.push({
            name,
            cat: category,
            ph: 'B',
            ts: Date.now() * 1000,
            pid: 1,
            tid: 1
        });
    }

    public endEvent(name: string, category: string = 'keras_to_onnx') {
        this.events.push({
            name,
            cat: category,
            ph: 'E',
            ts: Date.now() * 1000,
            pid: 1,
            tid: 1
        });
    }

    public recordMemory(bytes: number) {
        this.events.push({
            name: 'MemoryAlloc',
            cat: 'memory',
            ph: 'C',
            ts: Date.now() * 1000,
            pid: 1,
            tid: 1,
            args: { allocated: bytes }
        });
    }

    public save(path: string) {
        fs.writeFileSync(path, JSON.stringify(this.events, null, 2));
    }
}

export function validateMathematicalTolerance(
    kerasOutputs: Float32Array,
    onnxOutputs: Float32Array,
    tolerance: number = 1e-4
): boolean {
    if (kerasOutputs.length !== onnxOutputs.length) return false;
    for (let i = 0; i < kerasOutputs.length; i++) {
        if (Math.abs(kerasOutputs[i] - onnxOutputs[i]) > tolerance) {
            return false;
        }
    }
    return true;
}
