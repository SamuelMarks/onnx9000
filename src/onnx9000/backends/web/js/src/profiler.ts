/** Implementation details and semantic operations. */
export interface TraceEvent {
    name: string;
    ph: string;
    ts: number;
    pid: number;
    tid: number;
}

/** Implementation details and semantic operations. */

export interface TraceData {
    traceEvents: TraceEvent[];
}

/** Implementation details and semantic operations. */

export class Profiler {
    private events: TraceEvent[] = [];

    /** Implementation details and semantic operations. */

    public recordStart(name: string): void {
        this.events.push({ name, ph: 'B', ts: performance.now() * 1000, pid: 1, tid: 1 });
    }

    /** Implementation details and semantic operations. */

    public recordEnd(name: string): void {
        this.events.push({ name, ph: 'E', ts: performance.now() * 1000, pid: 1, tid: 1 });
    }

    /** Implementation details and semantic operations. */

    public dump(): string {
        return JSON.stringify({ traceEvents: this.events } as TraceData);
    }
}
