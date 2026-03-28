"""Module providing functionality for test_image."""

"""Test image."""


def test_image():
    """Docstring."""
    from onnx9000.genai.image import (
        ClassifierFreeGuidance,
        ControlNetSupport,
        DDIMScheduler,
        DiffusionMemoryOptimizer,
        DynamicResolutionScaler,
        EulerAncestralScheduler,
        HTMLCanvasExporter,
        ImageGeneratorParams,
        ImageToImage,
        Inpainting,
        LatentNoiseGenerator,
        LCMScheduler,
        MultiModelPipeline,
        NegativePromptHandler,
        PNDMScheduler,
        ProgressiveImageHooks,
        StableDiffusion1_5,
        StableDiffusionXL,
        UNetInference,
        VAEDecoder,
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
