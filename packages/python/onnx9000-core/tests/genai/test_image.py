import pytest
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


def test_image_params():
    params = ImageGeneratorParams("test")
    params.update(width=256)
    assert params.width == 256


def test_unet():
    unet = UNetInference()
    with pytest.raises(RuntimeError):
        unet.predict_noise([1.0], 1, [1.0])
    unet.load()
    assert unet.predict_noise([10.0], 1, [1.0]) == [1.0]


def test_vae():
    vae = VAEDecoder()
    assert vae.decode([0.18215]) == [255]


def test_ddim():
    ddim = DDIMScheduler(100)
    ddim.set_timesteps(10)
    assert len(ddim.timesteps) == 10
    assert ddim.step([0.1], 10, [0.5]) == [0.4]


def test_euler():
    euler = EulerAncestralScheduler()
    euler.set_timesteps(10)
    assert len(euler.sigmas) == 10
    assert euler.step([1.0], 10, [1.0]) == [0.5]


def test_pndm():
    pndm = PNDMScheduler()
    assert pndm.step([0.1], 10, [0.5]) == [0.5]
    for _ in range(5):
        pndm.step([0.1], 10, [0.5])
    assert len(pndm.ets) == 4


def test_lcm():
    lcm = LCMScheduler()
    lcm.set_timesteps(10)
    assert len(lcm.timesteps) == 10
    assert lcm.step([0.1], 10, [0.5]) == [0.4]


def test_cfg():
    cfg = ClassifierFreeGuidance(2.0)
    assert cfg.apply([1.0], [0.0]) == [2.0]


def test_negative_prompt():
    neg = NegativePromptHandler()
    neg.set_negative_prompt("bad")
    assert len(neg.get_embeddings()) == 768


def test_noise_generator():
    gen = LatentNoiseGenerator(42)
    noise = gen.generate((2, 2))
    assert len(noise) == 4


def test_multi_pipeline():
    pipe = MultiModelPipeline()
    pipe.add_model("test", None)
    assert pipe.run(1) == 1


def test_sd15():
    sd = StableDiffusion1_5()
    assert len(sd.generate("test")) == 192


def test_sdxl():
    sd = StableDiffusionXL()
    assert len(sd.generate("test")) == 384


def test_img2img():
    i2i = ImageToImage(0.5)
    assert i2i.process([100], "test") == [50]


def test_inpainting():
    inp = Inpainting()
    assert inp.process([100]) == [100]
    inp.set_mask([0, 1])
    assert inp.process([100, 100]) == [100, 255]


def test_controlnet():
    cn = ControlNetSupport()
    assert not cn.get_residuals()
    cn.set_control_image([1])
    assert len(cn.get_residuals()) == 320


def test_progressive_hooks():
    hooks = ProgressiveImageHooks()
    val = []
    hooks.register_hook(lambda s, l: val.append(s))
    hooks.trigger(1, [])
    assert val == [1]


def test_html_exporter():
    exp = HTMLCanvasExporter()
    assert "drawCanvas('sd-canvas', 512, 512);" in exp.export([], 512, 512)


def test_dynamic_scaler():
    scaler = DynamicResolutionScaler()
    assert scaler.scale(500, 500) == (448, 448)


def test_memory_opt():
    opt = DiffusionMemoryOptimizer()
    assert not opt.is_optimized()
    opt.enable_attention_slicing()
    assert opt.is_optimized()
