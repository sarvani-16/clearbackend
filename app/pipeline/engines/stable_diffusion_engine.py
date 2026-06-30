"""Stable Diffusion XL reconstruction engine — placeholder for satellite diffusion model."""

import numpy as np
from PIL import Image

from app.pipeline.engines.base import CloudReconstructionEngine, ReconstructionResult


class StableDiffusionReconstructionEngine(CloudReconstructionEngine):
    """
    Placeholder for a satellite-tuned Stable Diffusion XL inpainting pipeline.

    Real inference is not implemented. TODO markers indicate where the trained
    satellite diffusion model will be integrated.
    """

    def __init__(self):
        self._pipe = None
        self._device = None
        # TODO: Load satellite-tuned Stable Diffusion XL inpainting weights
        # self._pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        #     settings.SATELLITE_DIFFUSION_MODEL_PATH,
        #     torch_dtype=torch.float16,
        # )
        # self._device = resolve_torch_device()
        # self._pipe.to(self._device)

    @property
    def name(self) -> str:
        return "stable_diffusion"

    @property
    def badge(self) -> str:
        return "Stable Diffusion XL"

    def reconstruct(
        self,
        original_image: Image.Image,
        cloud_mask: np.ndarray,
        masked_image: Image.Image,
        *,
        source_filename: str = "",
    ) -> ReconstructionResult:
        # TODO: Preprocess inputs for the satellite diffusion model
        # - Resize to model-native resolution (e.g. 1024x1024)
        # - Normalize cloud_mask to single-channel binary mask
        # - Prepare masked_image as the conditioning input

        # TODO: Run satellite-tuned Stable Diffusion XL inpainting inference
        # result = self._pipe(
        #     prompt=SATELLITE_RECONSTRUCTION_PROMPT,
        #     negative_prompt=SATELLITE_NEGATIVE_PROMPT,
        #     image=masked_image,
        #     mask_image=cloud_mask,
        #     num_inference_steps=INFERENCE_STEPS,
        #     guidance_scale=GUIDANCE_SCALE,
        # ).images[0]

        # TODO: Post-process — composite only cloud-mask pixels from model output
        # cloud_free = composite_mask_regions(
        #     original=original_image,
        #     reconstructed=result,
        #     mask=cloud_mask,
        # )
        # cloud_free = cloud_free.resize(original_image.size, Image.Resampling.LANCZOS)

        raise NotImplementedError(
            "StableDiffusionReconstructionEngine is a placeholder. "
            "Integrate the trained satellite diffusion model at the TODO markers "
            "in stable_diffusion_engine.py before activating this engine."
        )
