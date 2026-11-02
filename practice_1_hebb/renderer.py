"""Render real letter images with PIL using system fonts.

Produces bipolar (+1 ink / -1 background) vectors at a fixed canvas size.
Used by the Hebbian demo to study associative memory on real letter
patterns instead of hand-coded 5x5 pixels.
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# A small, deterministic set of fonts found on CapyOS / Arch.
DEFAULT_FONTS = [
    "/usr/share/fonts/carlito/Carlito-Regular.ttf",
    "/usr/share/fonts/carlito/Carlito-Bold.ttf",
    "/usr/share/fonts/carlito/Carlito-Italic.ttf",
    "/usr/share/fonts/carlito/Carlito-BoldItalic.ttf",
]


def render_letter(ch: str, font_path: str, size: int = 28) -> np.ndarray:
    """Render `ch` centred on a `size x size` canvas and return a bipolar
    vector of length size*size: 1.0 for ink, -1.0 for background."""
    img = Image.new("L", (size, size), color=255)  # white = background
    draw = ImageDraw.Draw(img)
    # Pick a font size that fills ~85 % of the canvas; PIL's default
    # font-size unit is points; the exact value depends on the font.
    font = ImageFont.truetype(font_path, int(size * 1.6))
    draw.text((size // 2, size // 2), ch, fill=0, font=font, anchor="mm")
    arr = np.asarray(img, dtype=np.float64)
    return np.where(arr < 128, 1.0, -1.0).ravel()


def make_dataset(
    letters: str = "ABCDEFGHIJ",
    fonts: list[str] | None = None,
    size: int = 28,
    seed: int = 0,
) -> tuple[list[str], np.ndarray]:
    """Return (labels, patterns) for every (letter, font) pair.

    labels[i] is a string like "A:Carlito-Regular".
    patterns has shape (n_patterns, size*size).
    """
    if fonts is None:
        fonts = DEFAULT_FONTS
    fonts = [f for f in fonts if Path(f).exists()]
    if not fonts:
        raise FileNotFoundError("no usable TTF fonts found")
    rng = np.random.default_rng(seed)
    labels: list[str] = []
    rows: list[np.ndarray] = []
    for ch in letters:
        for fp in fonts:
            # Add a small random offset to make patterns more varied.
            img = Image.new("L", (size, size), 255)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(fp, int(size * 1.6))
            dx = int(rng.integers(-2, 3))
            dy = int(rng.integers(-2, 3))
            draw.text((size // 2 + dx, size // 2 + dy), ch,
                      fill=0, font=font, anchor="mm")
            arr = np.asarray(img, dtype=np.float64)
            rows.append(np.where(arr < 128, 1.0, -1.0).ravel())
            labels.append(f"{ch}:{Path(fp).stem}")
    return labels, np.stack(rows)


def occlude(pattern: np.ndarray, size: int, fraction: float = 0.25,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Zero out a `fraction` of pixels (set to -1, the background value).

    Returns a new pattern with a single rectangular block masked out — a
    more realistic test than bit-flip noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = pattern.copy().reshape(size, size)
    block_h = int(size * np.sqrt(fraction))
    block_w = int(size * np.sqrt(fraction))
    if block_h == 0 or block_w == 0:
        return out.ravel()
    top = int(rng.integers(0, size - block_h))
    left = int(rng.integers(0, size - block_w))
    out[top:top + block_h, left:left + block_w] = -1.0
    return out.ravel()


def noisy(pattern: np.ndarray, fraction: float = 0.2,
          rng: np.random.Generator | None = None) -> np.ndarray:
    """Flip a random `fraction` of pixels."""
    if rng is None:
        rng = np.random.default_rng()
    out = pattern.copy()
    mask = rng.random(out.size) < fraction
    out[mask] *= -1
    return out
