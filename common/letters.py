"""Shared letters dataset — the single source of images for all five practices.

The benchmark only means something when every network sees exactly the same
pictures and is scored with the same metric, so the dataset lives here and
every practice imports it.

Task: **find the letter X**. Uppercase letters A–Z are rendered with PIL from
real system fonts; a network must decide whether a picture shows "X" or some
other letter. Train and test use *different font families*, so the test set
really measures generalisation instead of memorisation.

Images are 28x28 binary (ink = 1.0, background = 0.0), flattened to 784
features. Augmentation (position jitter, glyph scaling, rotation) is applied
per sample so a font contributes many distinct pictures.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- constants
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TARGET_LETTER = "X"
SIZE = 28

# Train fonts: Carlito (metric-compatible with Calibri) + Liberation Sans.
TRAIN_FONTS = [
    "/usr/share/fonts/carlito/Carlito-Regular.ttf",
    "/usr/share/fonts/carlito/Carlito-Bold.ttf",
    "/usr/share/fonts/carlito/Carlito-Italic.ttf",
    "/usr/share/fonts/carlito/Carlito-BoldItalic.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Italic.ttf",
    "/usr/share/fonts/liberation/LiberationSans-BoldItalic.ttf",
]

# Test fonts: families the network has never seen (serif + freefont).
TEST_FONTS = [
    "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSerif-Italic.ttf",
    "/usr/share/fonts/gnu-free/FreeSans.otf",
    "/usr/share/fonts/gnu-free/FreeSansBold.otf",
    "/usr/share/fonts/gnu-free/FreeSansOblique.otf",
]

CACHE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "cache"


# ---------------------------------------------------------------- dataclass
@dataclass
class LetterSplit:
    """A ready-made train/test split of rendered letters."""

    X_train: np.ndarray          # (n_train, 784) float32, ink = 1.0
    y_train: np.ndarray          # (n_train,) int — index into `letters`
    X_test: np.ndarray           # (n_test, 784) float32
    y_test: np.ndarray           # (n_test,) int
    letters: str                # class alphabet, e.g. "ABC...Z"
    target: str                 # the letter we are hunting, e.g. "X"
    char_train: np.ndarray       # (n_train,) str — original character
    char_test: np.ndarray        # (n_test,) str
    font_train: np.ndarray       # (n_train,) str — font stem
    font_test: np.ndarray        # (n_test,) str

    @property
    def n_classes(self) -> int:
        return len(self.letters)

    @property
    def target_index(self) -> int:
        return self.letters.index(self.target)

    def binary_targets(self, split: str = "train") -> np.ndarray:
        """1.0 where the picture shows the target letter, else 0.0."""
        chars = self.char_train if split == "train" else self.char_test
        return (chars == self.target).astype(np.float32)

    def subset(self, split: str = "train", letters: str | None = None):
        """Return (X, y) restricted to the given letters."""
        X = self.X_train if split == "train" else self.X_test
        y = self.y_train if split == "train" else self.y_test
        chars = self.char_train if split == "train" else self.char_test
        if letters is None:
            return X, y
        keep = np.isin(chars, list(letters))
        return X[keep], y[keep]


# ---------------------------------------------------------------- rendering
def _glyph(ch: str, font_path: str) -> Image.Image:
    """Render `ch` big and crop to the ink bounding box (white background)."""
    font = ImageFont.truetype(font_path, 96)
    tmp = Image.new("L", (320, 320), color=255)
    ImageDraw.Draw(tmp).text((160, 160), ch, fill=0, font=font, anchor="mm")
    arr = np.asarray(tmp)
    ys, xs = np.where(arr < 128)
    if len(xs) == 0:                                   # missing glyph
        return tmp.crop((0, 0, 1, 1))
    return tmp.crop((int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1))


def render_letter(ch: str, font_path: str, size: int = SIZE,
                  box: float = 0.78, rng: np.random.Generator | None = None,
                  jitter: int = 2, angle: float = 8.0) -> np.ndarray:
    """Render one letter as a binary 28x28 vector (ink = 1.0).

    `box`  — how much of the canvas the glyph fills (0..1).
    `jitter` — random pixel offset of the glyph on the canvas.
    `angle` — random rotation in degrees (symmetric around 0).
    """
    if rng is None:
        rng = np.random.default_rng()
    g = _glyph(ch, font_path)
    w, h = g.size
    scale = (size * box) / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    g = g.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (size * 3, size * 3), color=255)
    dx = int(rng.integers(-jitter, jitter + 1))
    dy = int(rng.integers(-jitter, jitter + 1))
    canvas.paste(g, (size * 3 // 2 + dx - nw // 2, size * 3 // 2 + dy - nh // 2))

    if angle:
        canvas = canvas.rotate(float(rng.uniform(-angle, angle)),
                               resample=Image.BILINEAR, fillcolor=255,
                               center=(size * 3 // 2, size * 3 // 2))
    arr = np.asarray(canvas)[size:2 * size, size:2 * size]
    return (arr < 128).astype(np.float32).ravel()


def _font_stems(paths: list[str]) -> list[str]:
    return [Path(p).stem for p in paths]


# ------------------------------------------------------------------ builder
def build_letters(letters: str = LETTERS, size: int = SIZE,
                  n_train_per_font: int = 12, n_test_per_font: int = 4,
                  box: float = 0.78, jitter: int = 2, angle: float = 8.0,
                  seed: int = 0, use_cache: bool = True) -> LetterSplit:
    """Render the whole dataset and split it by font family.

    Train fonts and test fonts are disjoint, so test accuracy measures how
    well the principle generalises to *unseen handwriting styles*.
    """
    key = hashlib.md5(
        f"{letters}|{size}|{n_train_per_font}|{n_test_per_font}|{box}|"
        f"{jitter}|{angle}|{seed}|{','.join(TRAIN_FONTS)}|{','.join(TEST_FONTS)}"
        .encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"letters_{key}.npz"
    if use_cache and cache_path.exists():
        d = np.load(cache_path, allow_pickle=False)
        return LetterSplit(d["X_train"], d["y_train"], d["X_test"], d["y_test"],
                           str(d["letters"]), str(d["target"]),
                           d["char_train"], d["char_test"],
                           d["font_train"], d["font_test"])

    rng = np.random.default_rng(seed)
    Xtr, ytr, ctr, ftr = [], [], [], []
    Xte, yte, cte, fte = [], [], [], []

    for ch in letters:
        idx = letters.index(ch)
        for fp in TRAIN_FONTS:
            if not Path(fp).exists():
                continue
            for _ in range(n_train_per_font):
                Xtr.append(render_letter(ch, fp, size, box, rng, jitter, angle))
                ytr.append(idx)
                ctr.append(ch)
                ftr.append(Path(fp).stem)
        for fp in TEST_FONTS:
            if not Path(fp).exists():
                continue
            for _ in range(n_test_per_font):
                Xte.append(render_letter(ch, fp, size, box, rng, jitter, angle))
                yte.append(idx)
                cte.append(ch)
                fte.append(Path(fp).stem)

    split = LetterSplit(
        X_train=np.stack(Xtr), y_train=np.asarray(ytr, dtype=np.int64),
        X_test=np.stack(Xte), y_test=np.asarray(yte, dtype=np.int64),
        letters=letters, target=TARGET_LETTER,
        char_train=np.asarray(ctr), char_test=np.asarray(cte),
        font_train=np.asarray(ftr), font_test=np.asarray(fte),
    )
    if use_cache:
        np.savez_compressed(
            cache_path, X_train=split.X_train, y_train=split.y_train,
            X_test=split.X_test, y_test=split.y_test,
            letters=np.array(letters), target=np.array(TARGET_LETTER),
            char_train=split.char_train, char_test=split.char_test,
            font_train=split.font_train, font_test=split.font_test)
    return split


# ---------------------------------------------------------------- corruption
def add_noise(pattern: np.ndarray, fraction: float = 0.2,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Flip a random `fraction` of pixels (ink <-> background)."""
    if rng is None:
        rng = np.random.default_rng()
    out = np.asarray(pattern, dtype=np.float32).copy()
    mask = rng.random(out.size) < fraction
    out[mask] = 1.0 - out[mask]
    return out


def occlude(pattern: np.ndarray, size: int = SIZE, fraction: float = 0.25,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Mask out a rectangular block of the picture."""
    if rng is None:
        rng = np.random.default_rng()
    out = np.asarray(pattern, dtype=np.float32).copy().reshape(size, size)
    block = int(size * np.sqrt(fraction))
    if block == 0:
        return out.ravel()
    top = int(rng.integers(0, size - block))
    left = int(rng.integers(0, size - block))
    out[top:top + block, left:left + block] = 0.0
    return out.ravel()
