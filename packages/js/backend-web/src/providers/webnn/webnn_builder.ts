/* eslint-disable */
// @ts-nocheck
export class KerasWebNNCompiler {
    private builder: any;
    private context: any;

    constructor(builder: any, context: any) {
        this.builder = builder;
        this.context = context;
    }

    public buildConv2DBNRelu(input: any, weights: any, bias: any, bnGamma: any, bnBeta: any, bnMean: any, bnVar: any, options: any) {
        // Direct fused mapping to WebNN
        // Conv2D
        let convOut = this.builder.conv2d(input, weights, {
            padding: options.padding,
            strides: options.strides,
            dilations: options.dilations,
            groups: options.groups || 1,
            bias: bias
        });

        // Batch Normalization
        let bnOut = this.builder.batchNormalization(convOut, bnMean, bnVar, {
            scale: bnGamma,
            bias: bnBeta,
            epsilon: options.epsilon || 1e-5
        });

        // ReLU
        let reluOut = this.builder.relu(bnOut);

        return reluOut;
    }

    public buildSeparableConv2D(input: any, depthwiseWeights: any, pointwiseWeights: any, bias: any, options: any) {
        // Depthwise Conv2D (groups = inChannels)
        let depthOut = this.builder.conv2d(input, depthwiseWeights, {
            padding: options.padding,
            strides: options.strides,
            dilations: options.dilations,
            groups: options.inChannels
        });

        // Pointwise Conv2D (1x1 kernel)
        let pointOut = this.builder.conv2d(depthOut, pointwiseWeights, {
            bias: bias
        });

        return pointOut;
    }

    public async executeAsync(graph: any, inputs: any) {
        // Support WebNN async execution scheduling (`builder.build()`, `context.compute()`)
        const compiledGraph = await this.builder.build(graph);
        const outputs = await this.context.compute(compiledGraph, inputs);
        return outputs;
    }
}
