"""Demo reconstruction engine — returns pre-matched cloud-free reference imagery."""

import os

from PIL import Image

from app.config import settings
from app.pipeline.engines.base import CloudReconstructionEngine, ReconstructionResult


class DemoReconstructionEngine(CloudReconstructionEngine):
    """
    Demonstrates the full AI workflow without hallucinating terrain.

    Loads a matching cloud-free satellite image from demo_outputs/ or
    sample_outputs/ and returns it as the reconstruction result.
    """

    @property
    def name(self) -> str:
        return "demo"

    @property
    def badge(self) -> str:
        return "Demo AI Reconstruction"

    def reconstruct(
        self,
        original_image: Image.Image,
        cloud_mask,
        masked_image: Image.Image,
        *,
        source_filename: str = "",
    ) -> ReconstructionResult:
        clear_filename = self._resolve_clear_filename(source_filename)
        clear_path = self._resolve_clear_asset_path(clear_filename)

        cloud_free = Image.open(clear_path).convert("RGB")

        if cloud_free.size != original_image.size:
            cloud_free = cloud_free.resize(original_image.size, Image.Resampling.LANCZOS)

        print(f"[DemoReconstructionEngine] Loaded reference tile: {clear_path}")

        return ReconstructionResult(
            cloud_free_image=cloud_free,
            engine_name=self.name,
            engine_badge=self.badge,
        )

    @staticmethod
    def _resolve_clear_filename(source_filename: str) -> str:
        if "sample_cloudy_1" in source_filename:
            return "sample_clear_1.jpg"
        if "sample_cloudy_2" in source_filename:
            return "sample_clear_2.jpg"
        return "sample_clear_default.jpg"

    @staticmethod
    def _resolve_clear_asset_path(clear_filename: str) -> str:
        search_dirs = [
            settings.DEMO_OUTPUTS_FOLDER,
            settings.SAMPLE_OUTPUTS_FOLDER,
        ]

        for directory in search_dirs:
            candidate = os.path.join(directory, clear_filename)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"Cloud-free reference asset '{clear_filename}' not found in "
            f"demo_outputs/ or sample_outputs/."
        )
