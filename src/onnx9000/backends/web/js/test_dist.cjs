const index = require('./dist/index.js');
const assert = require('assert');

global.document = {
    getElementById: (id) => {
        if (id === 'missing') return null;
        return {
            innerHTML: '',
            id
        };
    }
};

global.performance = {
    now: () => Date.now()
};

async function runTests() {
    assert.ok(index.Tensor);
    assert.ok(index.env);
    assert.ok(index.Env);
    assert.ok(index.InferenceSession);
    assert.ok(index.ModelViewer);
    assert.ok(index.TensorInspector);
    assert.ok(index.Profiler);

    // 1. Env
    assert.strictEqual(index.env.logLevel, "warning");
    assert.strictEqual(index.Env.instance, index.env);

    // 2. Inspector
    assert.throws(() => new index.TensorInspector('missing'), /Container missing not found/);
    const inspector = new index.TensorInspector('found');
    const t = new index.Tensor("float32", [1, 2, 3]);
    inspector.inspect(t);
    
    // 3. Profiler
    const profiler = new index.Profiler();
    profiler.recordStart("test");
    profiler.recordEnd("test");
    const trace = JSON.parse(profiler.dump());
    assert.strictEqual(trace.traceEvents.length, 2);
    assert.strictEqual(trace.traceEvents[0].ph, 'B');
    assert.strictEqual(trace.traceEvents[1].ph, 'E');
    
    // 4. Session
    const session = await index.InferenceSession.create("dummy_path", { logSeverityLevel: 1 });
    const feeds = { input1: t };
    const result = await session.run(feeds, ["output1"], { extra: true });
    assert.deepStrictEqual(result, feeds);

    const session2 = await index.InferenceSession.create(new Uint8Array([1, 2, 3]));
    await session2.run({ i: t });

    // 5. Tensor
    const t1 = new index.Tensor("float32", new Float32Array([1, 2, 3, 4]), [2, 2]);
    assert.strictEqual(t1.size, 4);

    const t2 = new index.Tensor("int32", new Int32Array([1, 2]));
    assert.strictEqual(t2.size, 2);
    assert.deepStrictEqual(t2.dims, [2]);

    const t3 = new index.Tensor("string", ["a", "b"]);
    assert.strictEqual(t3.size, 2);

    const t4 = new index.Tensor("float32", [1.0, 2.0]);
    assert.ok(t4.data instanceof Float32Array);

    const t5 = new index.Tensor("int32", [1, 2]);
    assert.ok(t5.data instanceof Int32Array);

    const t6 = new index.Tensor("int64", [1, 2]);
    assert.ok(t6.data instanceof BigInt64Array);
    assert.strictEqual(t6.data[0], 1n);

    assert.throws(() => new index.Tensor("unsupported", [1, 2]), /Unsupported tensor type: unsupported/);
    assert.throws(() => new index.Tensor("float32", new Float32Array([1, 2]), [2, 2]), /Data size \(2\) does not match dimensions product \(4\)/);

    // 6. Viewer
    assert.throws(() => new index.ModelViewer('missing'), /Container missing not found/);
    const viewer = new index.ModelViewer('found2');
    viewer.render({ nodes: [{ id: "1" }] });
    viewer.render({});

    console.log("All tests passed!");
}

runTests().catch(e => {
    console.error(e);
    process.exit(1);
});
