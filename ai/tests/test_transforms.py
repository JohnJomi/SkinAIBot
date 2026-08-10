"""Training augments and varies; evaluation is fixed and deterministic."""

import torch
from PIL import Image

from ai.preprocessing.transforms import build_eval_transforms, build_train_transforms


def _sample_image(size=(160, 120)) -> Image.Image:
    generator = torch.Generator().manual_seed(3)
    pixels = torch.randint(0, 256, (size[1], size[0], 3), generator=generator)
    return Image.fromarray(pixels.numpy().astype("uint8"))


def test_both_pipelines_emit_the_configured_shape(config):
    image = _sample_image()
    expected = (3, config.image_size, config.image_size)

    for transform in (build_train_transforms(config), build_eval_transforms(config)):
        tensor = transform(image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == expected
        assert tensor.dtype == torch.float32


def test_eval_transform_is_deterministic(config):
    transform = build_eval_transforms(config)
    image = _sample_image()
    assert torch.equal(transform(image), transform(image))


def test_train_transform_augments(config):
    # A single pair can coincide - every augmentation here has some chance of
    # sampling the identity. Draw several and require that the transform did
    # not produce the same tensor every time; the seed keeps that decisive.
    transform = build_train_transforms(config)
    image = _sample_image()
    torch.manual_seed(0)

    first = transform(image)
    others = [transform(image) for _ in range(7)]

    assert any(not torch.equal(first, other) for other in others)


def test_normalization_is_applied(config):
    # A mid-grey image maps to a predictable value once standardised, which a
    # missing or misordered Normalize step would not produce.
    grey = Image.new("RGB", (200, 200), color=(128, 128, 128))
    tensor = build_eval_transforms(config)(grey)

    for channel, (mean, std) in enumerate(
        zip(config.normalization.mean, config.normalization.std, strict=True)
    ):
        expected = (128 / 255 - mean) / std
        assert torch.allclose(
            tensor[channel], torch.full_like(tensor[channel], expected), atol=1e-4
        )


def test_image_size_comes_from_config(config):
    resized = config.model_copy(update={"image_size": 64, "resize_size": 72})
    assert build_eval_transforms(resized)(_sample_image()).shape == (3, 64, 64)
