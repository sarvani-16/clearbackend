"""Future satellite reconstruction model — reserved placeholder implementation."""

import numpy as np
from PIL import Image

from app.pipeline.engines.base import CloudReconstructionEngine, ReconstructionResult


class FutureSatelliteModel(CloudReconstructionEngine):
    """
    Placeholder for a future proprietary satellite reconstruction model.

    This engine slot allows swapping in a custom-trained model without
    changing the pipeline orchestration or API surface.
    """

    def __init__(self):
        self._model = None
        # TODO: Load proprietary satellite reconstruction model weights
        # self._model = SatelliteReconstructionModel.from_checkpoint(
        #     settings.FUTURE_SATELLITE_MODEL_PATH,
        # )

    @property
    def name(self) -> str:
        return "future_satellite"

    @property
    def badge(self) -> str:
        return "Future Satellite Model"

    def reconstruct(
        self,
        original_image: Image.Image,
        cloud_mask: np.ndarray,
        masked_image: Image.Image,
        *,
        source_filename: str = "",
    ) -> ReconstructionResult:
        # TODO: Run proprietary satellite reconstruction inference
        # cloud_free_array = self._model.predict(
        #     image=np.array(original_image),
        #     mask=cloud_mask,
        # )
        # cloud_free = Image.fromarray(cloud_free_array)

        raise NotImplementedError(
            "FutureSatelliteModel is not yet available. "
            "Integrate the trained model at the TODO markers in "
            "future_satellite_model.py before activating this engine."
        )
