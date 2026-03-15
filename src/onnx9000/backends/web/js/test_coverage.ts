import * as index from "./src/index";
import { Env, env } from "./src/env";
import { TensorInspector } from "./src/inspector";
import { Profiler } from "./src/profiler";
import { InferenceSession } from "./src/session";
import { Tensor } from "./src/tensor";
import { ModelViewer } from "./src/viewer";

import * as assert from "node:assert";

// Mocking document and performance for browser-only code
globalThis.document = {
    getElementById: (id: string) => {
        /** Retrieves a mock DOM element or returns null if the id is missing to simulate browser execution context. */
        if(id === 'missing') return null;
        return {
            innerHTML: '',
            id
        };
    }
} as any;

globalThis.performance = {
    now: () => Date.now()
} as any;

async function runTests() {
    assert.ok(index.Tensor);
    assert.ok(index.env);
    assert.ok(index.Env);
    assert.ok(index.InferenceSession);
    assert.ok(index.ModelViewer);
    assert.ok(index.TensorInspector);
    assert.ok(index.Profiler);

    // 1. Env
    assert.strictEqual(env.logLevel, "warning");
    assert.strictEqual(Env.instance, env);

    // 2. Inspector
    assert.throws(() => new TensorInspector('missing'), /Container missing not found/);
    const inspector = new TensorInspector('found');
    const t = new Tensor("float32", [1, 2, 3]);
    inspector.inspect(t);
    // document mock innerHTML will be set
    
    // 3. Profiler
    const profiler = new Profiler();
    profiler.recordStart("test");
    profiler.recordEnd("test");
    const trace = JSON.parse(profiler.dump());
    assert.strictEqual(trace.traceEvents.length, 2);
    assert.strictEqual(trace.traceEvents[0].ph, 'B');
    assert.strictEqual(trace.traceEvents[1].ph, 'E');
    
    // 4. Session
    const session = await InferenceSession.create("dummy_path", { logSeverityLevel: 1 });
    const feeds = { input1: t };
    const result = await session.run(feeds, ["output1"], { extra: true });
    assert.deepStrictEqual(result, feeds);

    const session2 = await InferenceSession.create(new Uint8Array([1, 2, 3]));
    await session2.run({ i: t });

    // 5. Tensor
    // 5.1 With dims
    const t1 = new Tensor("float32", new Float32Array([1, 2, 3, 4]), [2, 2]);
    assert.strictEqual(t1.size, 4);

    // 5.2 Without dims
    const t2 = new Tensor("int32", new Int32Array([1, 2]));
    assert.strictEqual(t2.size, 2);
    assert.deepStrictEqual(t2.dims, [2]);

    // 5.3 String array
    const t3 = new Tensor("string", ["a", "b"]);
    assert.strictEqual(t3.size, 2);

    // 5.4 Float32 array creation
    const t4 = new Tensor("float32", [1.0, 2.0]);
    assert.ok(t4.data instanceof Float32Array);

    // 5.5 Int32 array creation
    const t5 = new Tensor("int32", [1, 2]);
    assert.ok(t5.data instanceof Int32Array);

    // 5.6 Int64 array creation
    const t6 = new Tensor("int64", [1, 2]);
    assert.ok(t6.data instanceof BigInt64Array);
    assert.strictEqual(t6.data[0], 1n);

    // 5.7 Unsupported type
    assert.throws(() => new Tensor("unsupported", [1, 2]), /Unsupported tensor type: unsupported/);

    // 5.8 Size mismatch
    assert.throws(() => new Tensor("float32", new Float32Array([1, 2]), [2, 2]), /Data size \(2\) does not match dimensions product \(4\)/);

    // 6. Viewer
    assert.throws(() => new ModelViewer('missing'), /Container missing not found/);
    const viewer = new ModelViewer('found2');
    viewer.render({ nodes: [{ id: "1" }] });
    viewer.render({});

    console.log("All tests passed!");
}

await runTests();
