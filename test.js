import { KerasPythonParser } from './apps/sphinx-demo-ui/src/core/KerasPythonParser';
async function main() {
    const pythonCode = `import keras
from keras import layers, models

model = models.Sequential([
    layers.Input(shape=(28, 28, 1), name='image_input'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax', name='output_probs')
])
`;
    // We can't really run pyodide easily here in node without node packages, 
    // but wait, KerasPythonParser.js loads pyodide from cdn!
    try {
        const result = await KerasPythonParser.js.parse(pythonCode);
        console.log(JSON.stringify(result, null, 2));
    }
    catch (e) {
        console.error(e);
    }
}
main().catch(console.error);
