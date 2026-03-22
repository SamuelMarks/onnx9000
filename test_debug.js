const { FlatBufferBuilder } = require('./packages/js/tflite-exporter/dist/flatbuffer/builder.js');
const { FlatBufferReader } = require('./packages/js/tflite-exporter/dist/flatbuffer/reader.js');
const { TFLiteExporter } = require('./packages/js/tflite-exporter/dist/exporter.js');
const { compileGraphToTFLite } = require('./packages/js/tflite-exporter/dist/compiler/subgraph.js');
const { Graph, Tensor } = require('./packages/js/core/dist/ir/graph.js');

const exporter = new TFLiteExporter();
const graph = new Graph('TestGraph');
const subgraphsOffset = compileGraphToTFLite(graph, exporter);
exporter.builder.startVector(4, 1, 4);
exporter.builder.addOffset(subgraphsOffset);
const subgraphsVecOffset = exporter.builder.endVector();
const buf = exporter.finish(subgraphsVecOffset, 'test_graph_compilation');

const reader = new FlatBufferReader(buf);
console.log('Root offset:', reader.getRoot());
