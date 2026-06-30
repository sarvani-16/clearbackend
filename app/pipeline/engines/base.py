"""Abstract interface for modular cloud reconstruction engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ReconstructionResult:
    """Output from a cloud reconstruction engine."""

    cloud_free_image: Image.Image
    engine_name: str
    engine_badge: str


class CloudReconstructionEngine(ABC):
    """
    Interface for AI cloud reconstruction backends.

    Each implementation receives the original satellite image, a binary cloud
    mask, and a version of the image with cloud pixels removed. Implementations
    must not modify pixels outside the cloud mask unless returning a full
    pre-matched reference image (demo mode).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Internal engine identifier."""

    @property
    @abstractmethod
    def badge(self) -> str:
        """Human-readable badge label for the frontend."""

    @abstractmethod
    def reconstruct(
        self,
        original_image: Image.Image,
        cloud_mask: np.ndarray,
        masked_image: Image.Image,
        *,
        source_filename: str = "",
    ) -> ReconstructionResult:
        """
        Reconstruct cloud-free satellite imagery.

        Args:
            original_image: Full RGB satellite tile before reconstruction.
            cloud_mask: Binary mask where non-zero pixels indicate clouds.
            masked_image: Original image with cloud pixels removed (zeroed).
            source_filename: Original upload filename for demo asset matching.

        Returns:
            ReconstructionResult containing the cloud-free image and engine metadata.
        """
