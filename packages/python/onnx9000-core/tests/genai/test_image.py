def test_image():
    from onnx9000.genai.image import (
        ImageGeneratorParams,
        UNetInference,
        VAEDecoder,
        DDIMScheduler,
        EulerAncestralScheduler,
        PNDMScheduler,
        LCMScheduler,
        ClassifierFreeGuidance,
        NegativePromptHandler,
        LatentNoiseGenerator,
        MultiModelPipeline,
        StableDiffusion1_5,
        StableDiffusionXL,
        ImageToImage,
        Inpainting,
        ControlNetSupport,
        ProgressiveImageHooks,
        HTMLCanvasExporter,
        DynamicResolutionScaler,
        DiffusionMemoryOptimizer,
    )

    assert ImageGeneratorParams()._initialized
    assert UNetInference()._initialized
    assert VAEDecoder()._initialized
    assert DDIMScheduler()._initialized
    assert EulerAncestralScheduler()._initialized
    assert PNDMScheduler()._initialized
    assert LCMScheduler()._initialized
    assert ClassifierFreeGuidance()._initialized
    assert NegativePromptHandler()._initialized
    assert LatentNoiseGenerator()._initialized
    assert MultiModelPipeline()._initialized
    assert StableDiffusion1_5()._initialized
    assert StableDiffusionXL()._initialized
    assert ImageToImage()._initialized
    assert Inpainting()._initialized
    assert ControlNetSupport()._initialized
    assert ProgressiveImageHooks()._initialized
    assert HTMLCanvasExporter()._initialized
    assert DynamicResolutionScaler()._initialized
    assert DiffusionMemoryOptimizer()._initialized
