"""Factory for selecting the active cloud reconstruction engine at runtime."""

import os

from app.pipeline.engines.base import CloudReconstructionEngine
from app.pipeline.engines.demo_engine import DemoReconstructionEngine
from app.pipeline.engines.future_satellite_model import FutureSatelliteModel
from app.pipeline.engines.stable_diffusion_engine import StableDiffusionReconstructionEngine

_ENGINE_REGISTRY: dict[str, type[CloudReconstructionEngine]] = {
    "demo": DemoReconstructionEngine,
    "stable_diffusion": StableDiffusionReconstructionEngine,
    "future_satellite": FutureSatelliteModel,
}


def create_reconstruction_engine() -> CloudReconstructionEngine:
    """
    Instantiate the single active reconstruction engine for this process.

  Selection is controlled by the RECONSTRUCTION_ENGINE environment variable:
    - demo              → DemoReconstructionEngine (default)
    - stable_diffusion  → StableDiffusionReconstructionEngine
    - future_satellite  → FutureSatelliteModel
    """
    engine_key = os.getenv("RECONSTRUCTION_ENGINE", "demo").lower().strip()

    engine_cls = _ENGINE_REGISTRY.get(engine_key)
    if engine_cls is None:
        print(
            f"[EngineFactory] Unknown RECONSTRUCTION_ENGINE='{engine_key}'. "
            "Falling back to DemoReconstructionEngine."
        )
        engine_cls = DemoReconstructionEngine

    engine = engine_cls()
    print(f"[EngineFactory] Active engine: {engine.name} ({engine.badge})")
    return engine
