const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    // Step 363: Puppeteer for end-to-end browser execution tests
    const browser = await puppeteer.launch({ headless: true });
    const page = await browser.newPage();
    
    // Inject the UMD bundle
    const bundlePath = path.join(__dirname, 'dist', 'onnx9000-web.min.js');
    await page.addScriptTag({ path: bundlePath });

    const result = await page.evaluate(async () => {
        // Step 362: Basic test logic executed in headless Chrome context
        const session = await window.onnx9000.InferenceSession.create("dummy");
        const t = new window.onnx9000.Tensor("float32", [1, 2, 3]);
        const output = await session.run({ "a": t });
        
        return {
            tensorSize: t.size,
            outputSize: output["a"].size,
            isInstanceOf: session instanceof window.onnx9000.InferenceSession
        };
    });

    console.log("E2E Result:", result);

    if (result.tensorSize !== 3 || !result.isInstanceOf) {
        console.error("Test failed!");
        process.exit(1);
    }
    
    console.log("All E2E tests passed.");
    await browser.close();
})();
