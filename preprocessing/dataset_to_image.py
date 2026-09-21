from pathlib import Path

from benchmark_data.schema import BenchmarkSample
from preprocessing.manifest import VisualSample
from preprocessing.text_to_image import TextToImageRenderer


class DatasetImagePreprocessor:
    """
    Mengubah BenchmarkSample menjadi VisualSample.

    Alur:

        BenchmarkSample
              ↓
        TextToImageRenderer
              ↓
            PNG
              ↓
        VisualSample
    """

    def __init__(
        self,
        renderer: TextToImageRenderer,
        output_dir: Path,
    ):
        self.renderer = renderer
        self.output_dir = Path(output_dir)

    def _get_rendering_config(self) -> dict:
        return {
            "width": self.renderer.width,
            "height": self.renderer.height,
            "background": self.renderer.background,
            "text_color": self.renderer.text_color,
            "margin": self.renderer.margin,
            "font_size": self.renderer.font_size,
            "line_spacing": self.renderer.line_spacing,
            "font_path": (
                str(self.renderer.font_path)
                if self.renderer.font_path is not None
                else None
            ),
        }

    def process_sample(
        self,
        sample: BenchmarkSample,
    ) -> VisualSample:
        benchmark_dir = (
            self.output_dir / sample.benchmark
        )

        image_path = (
            benchmark_dir
            / f"{sample.sample_id}.png"
        )

        self.renderer.render(
            text=sample.prompt,
            output_path=image_path,
        )

        return VisualSample(
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
            image_path=image_path,
            source_prompt=sample.prompt,
            rendering_config=self._get_rendering_config(),
        )