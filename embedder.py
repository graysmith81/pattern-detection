"""
embedder.py — Zero-shot CNN embedding and retrieval for chart pattern matching.

Uses a pretrained ResNet50 or EfficientNet-B0 backbone (ImageNet weights, no
fine-tuning) as a feature extractor.  Charts are cropped using the same
percentages as engine.py, resized to 224×224, and embedded into an
L2-normalised vector.  Vectors are stored in a simple .npz index on disk so
charts do not need to be re-embedded on every session.

Graceful degradation
--------------------
If torch / torchvision are not installed the module still imports without
error.  TORCH_AVAILABLE is False, and calling embed_image() or any index
function that requires embedding raises ImportError with a helpful message.

Public API
----------
    TORCH_AVAILABLE : bool
    load_model(backbone="resnet50") -> nn.Module
    crop_chart(img_bgr, ...) -> np.ndarray          # uint8 RGB 224×224
    embed_image(img_bgr, model) -> np.ndarray       # float32 L2-normed vector
    ChartIndex                                       # index dataclass
    build_index(files, model) -> ChartIndex
    save_index(index, path)
    load_index(path) -> ChartIndex
    query_index(index, query_vec, top_n, threshold) -> list[EmbedResult]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# Reuse filename parsing from engine — no circular dependency
from engine import parse_filename_metadata, image_from_bytes

# ─────────────────────────────────────────────────────────────────────────────
#  Crop constants — kept in sync with engine.py
# ─────────────────────────────────────────────────────────────────────────────

TOP_CROP_PCT:    float = 0.04
BOTTOM_CROP_PCT: float = 0.08
RIGHT_CROP_PCT:  float = 0.09
LEFT_CROP_PCT:   float = 0.00   # engine does not crop the left edge

EMBED_SIZE: int = 224           # canonical input resolution for both backbones

# ─────────────────────────────────────────────────────────────────────────────
#  Optional PyTorch import
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as T
    from PIL import Image as PILImage

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbedResult:
    """Single retrieval result returned by query_index()."""
    chart_path: str
    similarity: float       # cosine similarity in [0, 1]
    symbol:     str = ""
    timeframe:  str = ""
    datetime:   str = ""


@dataclass
class ChartIndex:
    """
    In-memory embedding index.

    vectors    : float32 array of shape (N, D) — one L2-normed row per chart
    filenames  : list of N original filenames / paths
    symbols    : list of N symbol strings (may be empty)
    timeframes : list of N timeframe strings (may be empty)
    datetimes  : list of N datetime strings (may be empty)
    backbone   : name of the backbone used to build this index
    """
    vectors:    np.ndarray
    filenames:  List[str] = field(default_factory=list)
    symbols:    List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    datetimes:  List[str] = field(default_factory=list)
    backbone:   str       = "resnet50"

    @property
    def size(self) -> int:
        return len(self.filenames)


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache so the model is loaded only once per session
_model_cache: dict = {}


def load_model(backbone: str = "resnet50"):
    """
    Load a pretrained backbone with its classification head removed.

    backbone : "resnet50"        → 2048-dim embeddings
               "efficientnet_b0" → 1280-dim embeddings

    The loaded model is cached; subsequent calls with the same name are free.
    Raises ImportError if torch/torchvision are not installed.
    """
    _require_torch()

    if backbone in _model_cache:
        return _model_cache[backbone]

    if backbone == "resnet50":
        weights = tv_models.ResNet50_Weights.DEFAULT
        base    = tv_models.resnet50(weights=weights)
        # Drop the final FC layer; output is the 2048-d avg-pool feature map
        model = nn.Sequential(*list(base.children())[:-1])

    elif backbone == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT
        base    = tv_models.efficientnet_b0(weights=weights)
        # Keep features + adaptive avg-pool, drop classifier
        model = nn.Sequential(base.features, base.avgpool)

    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            "Choose 'resnet50' or 'efficientnet_b0'."
        )

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    _model_cache[backbone] = model
    return model


def _get_transform():
    """Standard ImageNet normalisation transform for 224×224 RGB input."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Chart crop and embed
# ─────────────────────────────────────────────────────────────────────────────

def crop_chart(
    img_bgr:         np.ndarray,
    top_crop_pct:    float = TOP_CROP_PCT,
    bottom_crop_pct: float = BOTTOM_CROP_PCT,
    right_crop_pct:  float = RIGHT_CROP_PCT,
    left_crop_pct:   float = LEFT_CROP_PCT,
    output_size:     int   = EMBED_SIZE,
) -> np.ndarray:
    """
    Crop axes/labels from a BGR chart image and resize to a square RGB canvas.

    The crop percentages mirror engine.py so both engines look at the same
    chart region.  The result is a uint8 RGB array of shape
    (output_size, output_size, 3) ready for the CNN transform.

    Resizing to a fixed square makes embeddings scale-, magnitude-, and
    time-stretch-invariant: whatever the original aspect ratio or price range,
    the shape fills the 224×224 frame.
    """
    import cv2

    h, w = img_bgr.shape[:2]
    tc = int(h * top_crop_pct)
    bc = int(h * bottom_crop_pct)
    lc = int(w * left_crop_pct)
    rc = int(w * right_crop_pct)

    y0, y1 = tc, max(tc + 1, h - bc)
    x0, x1 = lc, max(lc + 1, w - rc)

    crop    = img_bgr[y0:y1, x0:x1]
    resized = cv2.resize(crop, (output_size, output_size),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def embed_image(img_bgr: np.ndarray, model) -> np.ndarray:
    """
    Produce an L2-normalised embedding vector from a BGR chart image.

    Pipeline:
      1. Crop the chart area (strip axes/labels)
      2. Resize to 224×224 RGB
      3. Apply ImageNet normalisation
      4. Forward pass through the backbone (torch.no_grad)
      5. Flatten and L2-normalise

    Returns a float32 numpy array of shape (D,).
    Raises ImportError if torch/torchvision are not installed.
    """
    _require_torch()

    rgb_crop  = crop_chart(img_bgr)
    pil_img   = PILImage.fromarray(rgb_crop)
    transform = _get_transform()
    tensor    = transform(pil_img).unsqueeze(0)     # (1, 3, 224, 224)

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        features = model(tensor)                    # (1, D, 1, 1) or (1, D)

    vec = features.squeeze().cpu().numpy().astype(np.float32)

    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm

    return vec


# ─────────────────────────────────────────────────────────────────────────────
#  Index construction
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    image_files: list,
    model,
    backbone: str = "resnet50",
    progress_callback=None,
) -> ChartIndex:
    """
    Embed a collection of chart images and return a ChartIndex.

    Parameters
    ----------
    image_files : list of (filename: str, data: bytes | np.ndarray)
        Each entry is a (name, data) pair.  data may be raw bytes or a BGR
        numpy array.
    model : loaded backbone from load_model()
    backbone : backbone name stored in the index for later validation
    progress_callback : optional callable(i: int, total: int, name: str)
        Called before embedding each image; useful for Streamlit progress bars.

    Returns a ChartIndex ready for querying or saving.
    Corrupt/unreadable images are skipped silently.
    """
    _require_torch()

    vectors:    list[np.ndarray] = []
    filenames:  list[str]        = []
    symbols:    list[str]        = []
    timeframes: list[str]        = []
    datetimes:  list[str]        = []
    total = len(image_files)

    for i, (name, data) in enumerate(image_files):
        if progress_callback:
            progress_callback(i, total, name)

        try:
            img_bgr = image_from_bytes(data) if isinstance(data, bytes) else data
            if img_bgr is None:
                continue

            vec  = embed_image(img_bgr, model)
            meta = parse_filename_metadata(name)

            vectors.append(vec)
            filenames.append(name)
            symbols.append(meta.get("symbol",    ""))
            timeframes.append(meta.get("timeframe", ""))
            datetimes.append(meta.get("datetime",  ""))

        except Exception:
            continue     # skip unreadable images without crashing the build

    if not vectors:
        dummy_dim = 2048 if backbone == "resnet50" else 1280
        return ChartIndex(
            vectors=np.empty((0, dummy_dim), dtype=np.float32),
            backbone=backbone,
        )

    return ChartIndex(
        vectors=np.vstack(vectors).astype(np.float32),
        filenames=filenames,
        symbols=symbols,
        timeframes=timeframes,
        datetimes=datetimes,
        backbone=backbone,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_index(index: ChartIndex, path: str | Path) -> None:
    """
    Persist a ChartIndex to a compressed .npz file.

    The file can be reloaded with load_index() to avoid re-embedding on the
    next session.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        vectors    = index.vectors,
        filenames  = np.array(index.filenames,  dtype=object),
        symbols    = np.array(index.symbols,    dtype=object),
        timeframes = np.array(index.timeframes, dtype=object),
        datetimes  = np.array(index.datetimes,  dtype=object),
        backbone   = np.array([index.backbone], dtype=object),
    )


def load_index(path: str | Path) -> ChartIndex:
    """
    Load a ChartIndex previously saved with save_index().

    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding index not found: {path}")

    data = np.load(path, allow_pickle=True)

    return ChartIndex(
        vectors    = data["vectors"].astype(np.float32),
        filenames  = data["filenames"].tolist(),
        symbols    = data["symbols"].tolist(),
        timeframes = data["timeframes"].tolist(),
        datetimes  = data["datetimes"].tolist(),
        backbone   = str(data["backbone"][0]),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Querying
# ─────────────────────────────────────────────────────────────────────────────

def query_index(
    index:     ChartIndex,
    query_vec: np.ndarray,
    top_n:     int   = 10,
    threshold: float = 0.0,
) -> list[EmbedResult]:
    """
    Rank all indexed charts by cosine similarity to query_vec.

    Because both index rows and query_vec are L2-normalised, cosine similarity
    reduces to a single matrix–vector dot product — fast even for large indexes.

    Parameters
    ----------
    index     : a ChartIndex built or loaded with build_index / load_index
    query_vec : L2-normalised float32 vector of the same dimension as index.vectors
    top_n     : maximum number of results to return
    threshold : minimum cosine similarity; results below this are dropped

    Returns a list of EmbedResult sorted by similarity descending.
    """
    if index.size == 0:
        return []

    # Re-normalise defensively in case the caller skipped it
    norm = np.linalg.norm(query_vec)
    if norm > 1e-8:
        query_vec = query_vec / norm

    scores = index.vectors @ query_vec.astype(np.float32)   # (N,)
    scores = np.clip(scores, 0.0, 1.0)                      # numerical safety

    order = np.argsort(scores)[::-1]

    results = []
    for idx in order[:top_n]:
        sim = float(scores[idx])
        if sim < threshold:
            break
        results.append(EmbedResult(
            chart_path = index.filenames[idx],
            similarity = sim,
            symbol     = index.symbols[idx]    if index.symbols    else "",
            timeframe  = index.timeframes[idx] if index.timeframes else "",
            datetime   = index.datetimes[idx]  if index.datetimes  else "",
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch and torchvision are required for CNN embedding but are not "
            "installed.\n\nInstall them with:\n"
            "    pip install torch torchvision\n"
            "or visit https://pytorch.org/get-started/locally/ for platform-specific instructions."
        )
