from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class VisualSample:
    """
    Representasi visual dari satu BenchmarkSample.

    VisualSample hanya menyimpan informasi mengenai
    hasil preprocessing dan identitas sample.

    Data eksperimen model tidak disimpan di sini.
    """

    sample_id: str
    benchmark: str
    image_path: Path
    source_prompt: Optional[str] = None
    rendering_config: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "benchmark": self.benchmark,
            "image_path": str(self.image_path),
            "source_prompt": self.source_prompt,
            "rendering_config": self.rendering_config,
        }