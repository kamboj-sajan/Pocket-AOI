"""Image loading and saving. Resize and save helpers arrive in Block 1."""

from pathlib import Path

from PIL import Image, ImageOps


def load_rgb(path: str | Path) -> Image.Image:
    """Open a photo as RGB, upright: phones store rotation in EXIF rather than in the pixels."""
    # A with-block is Python's RAII: the file closes when the block ends.
    with Image.open(path) as photo:
        return ImageOps.exif_transpose(photo).convert("RGB")
