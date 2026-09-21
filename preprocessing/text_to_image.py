from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


class TextToImageRenderer:
    """
    Renderer deterministik untuk mengubah teks menjadi gambar.

    Tujuan:
        BenchmarkSample.prompt
            ↓
        TextToImageRenderer
            ↓
        PNG

    Renderer tidak mengubah isi semantik prompt.
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        background: str = "white",
        text_color: str = "black",
        margin: int = 64,
        font_path: Optional[Path] = None,
        font_size: int = 32,
        line_spacing: int = 10,
    ):
        if width <= 0:
            raise ValueError("width harus > 0")

        if height <= 0:
            raise ValueError("height harus > 0")

        if margin < 0:
            raise ValueError("margin tidak boleh negatif")

        if font_size <= 0:
            raise ValueError("font_size harus > 0")

        if line_spacing < 0:
            raise ValueError("line_spacing tidak boleh negatif")

        if margin * 2 >= width:
            raise ValueError("margin terlalu besar terhadap width")

        if margin * 2 >= height:
            raise ValueError("margin terlalu besar terhadap height")

        self.width = width
        self.height = height
        self.background = background
        self.text_color = text_color
        self.margin = margin
        self.font_path = font_path
        self.font_size = font_size
        self.line_spacing = line_spacing

    def _load_font(self):
        """
        Memuat font.

        Jika font_path tidak diberikan, gunakan font default
        dari Pillow.
        """
        if self.font_path is None:
            return ImageFont.load_default()

        font_path = Path(self.font_path)

        if not font_path.exists():
            raise FileNotFoundError(
                f"Font tidak ditemukan: {font_path}"
            )

        return ImageFont.truetype(
            str(font_path),
            self.font_size,
        )

    def _wrap_line(
        self,
        draw: ImageDraw.ImageDraw,
        line: str,
        font,
        max_width: int,
    ) -> list[str]:
        """
        Membungkus satu baris teks agar tidak melebihi max_width.
        """

        if not line:
            return [""]

        words = line.split()

        if not words:
            return [""]

        wrapped = []
        current = ""

        for word in words:
            candidate = word if not current else f"{current} {word}"

            bbox = draw.textbbox(
                (0, 0),
                candidate,
                font=font,
            )

            candidate_width = bbox[2] - bbox[0]

            if candidate_width <= max_width:
                current = candidate
                continue

            if current:
                wrapped.append(current)

            # Jika satu kata sendiri terlalu panjang,
            # pecah berdasarkan karakter.
            word_bbox = draw.textbbox(
                (0, 0),
                word,
                font=font,
            )

            word_width = word_bbox[2] - word_bbox[0]

            if word_width <= max_width:
                current = word
                continue

            current = ""

            partial = ""

            for char in word:
                candidate_char = partial + char

                char_bbox = draw.textbbox(
                    (0, 0),
                    candidate_char,
                    font=font,
                )

                char_width = char_bbox[2] - char_bbox[0]

                if char_width <= max_width:
                    partial = candidate_char
                else:
                    if partial:
                        wrapped.append(partial)

                    partial = char

            current = partial

        if current:
            wrapped.append(current)

        return wrapped

    def _prepare_lines(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font,
        max_width: int,
    ) -> list[str]:
        """
        Mempertahankan newline asli dan melakukan wrapping
        pada masing-masing baris.
        """

        lines = []

        for original_line in text.splitlines():
            wrapped = self._wrap_line(
                draw=draw,
                line=original_line,
                font=font,
                max_width=max_width,
            )

            lines.extend(wrapped)

        if not lines:
            lines.append("")

        return lines

    def render(
        self,
        text: str,
        output_path: Path,
    ) -> Path:
        """
        Render text menjadi PNG.
        """

        if not isinstance(text, str):
            raise TypeError("text harus berupa string")

        if not text.strip():
            raise ValueError("text tidak boleh kosong")

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image = Image.new(
            "RGB",
            (self.width, self.height),
            self.background,
        )

        draw = ImageDraw.Draw(image)
        font = self._load_font()

        max_width = self.width - (2 * self.margin)

        lines = self._prepare_lines(
            draw=draw,
            text=text,
            font=font,
            max_width=max_width,
        )

        # Hitung tinggi setiap baris.
        line_heights = []

        for line in lines:
            bbox = draw.textbbox(
                (0, 0),
                line if line else "Ag",
                font=font,
            )

            height = bbox[3] - bbox[1]
            line_heights.append(height)

        total_height = (
            sum(line_heights)
            + self.line_spacing * max(0, len(lines) - 1)
        )

        max_height = self.height - (2 * self.margin)

        if total_height > max_height:
            raise ValueError(
                "Teks tidak dapat dimuat ke dalam gambar "
                f"{self.width}x{self.height}. "
                f"Required height={total_height}, "
                f"available height={max_height}."
            )

        y = self.margin

        for line, line_height in zip(lines, line_heights):
            draw.text(
                (self.margin, y),
                line,
                fill=self.text_color,
                font=font,
            )

            y += line_height + self.line_spacing

        image.save(
            output_path,
            format="PNG",
        )

        return output_path