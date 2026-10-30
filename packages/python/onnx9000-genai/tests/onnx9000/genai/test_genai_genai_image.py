"""Unit tests for image generation and diffusion schedulers in `onnx9000.genai.image`."""

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


def test_image_diffusion_pipeline_and_schedulers() -> None:
    """Verify ImageGeneratorParams, UNetInference, VAEDecoder, Schedulers, CFG, SD1.5, SDXL, ControlNet, and exporters."""
    params = ImageGeneratorParams("a cat")
    params.update(height=256)
    assert params.height == 256

    unet = UNetInference()
    unet.load()
    assert unet.predict_noise([10.0], 1, [1.0]) == [1.0]

    vae = VAEDecoder()
    assert vae.decode([0.18215]) == [255]

    ddim = DDIMScheduler(100)
    ddim.set_timesteps(5)
    assert ddim.step([0.1], 1, [0.5]) == [0.4]

    euler = EulerAncestralScheduler()
    euler.set_timesteps(5)
    assert euler.step([1.0], 1, [1.0]) == [0.5]

    pndm = PNDMScheduler()
    assert pndm.step([0.1], 1, [0.5]) == [0.5]

    lcm = LCMScheduler()
    lcm.set_timesteps(4)
    assert lcm.step([0.1], 1, [0.5]) == [0.4]

    cfg = ClassifierFreeGuidance(2.0)
    assert cfg.apply([1.0], [0.0]) == [2.0]

    neg = NegativePromptHandler()
    neg.set_negative_prompt("blurry")
    assert len(neg.get_embeddings()) == 768

    rng = LatentNoiseGenerator(7)
    assert len(rng.generate((2, 2))) == 4

    pipe = MultiModelPipeline()
    pipe.add_model("vae", vae)
    assert pipe.run(42) == 42

    assert len(StableDiffusion1_5().generate("cat")) == 192
    assert len(StableDiffusionXL().generate("cat")) == 384

    assert ImageToImage(0.5).process([100], "cat") == [50]

    inp = Inpainting()
    inp.set_mask([0, 1])
    assert inp.process([100, 100]) == [100, 255]

    cn = ControlNetSupport()
    cn.set_control_image([1])
    assert len(cn.get_residuals()) == 320

    hooks = ProgressiveImageHooks()
    steps: list[int] = []
    hooks.register_hook(lambda s, _l: steps.append(s))
    hooks.trigger(3, [])
    assert steps == [3]

    assert "sd-canvas" in HTMLCanvasExporter().export([], 64, 64)
    assert DynamicResolutionScaler().scale(500, 500) == (448, 448)

    opt = DiffusionMemoryOptimizer()
    opt.enable_attention_slicing()
    assert opt.is_optimized() is True
