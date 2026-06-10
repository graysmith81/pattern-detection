"""
matcher.py — Visual sliding-window chart pattern matcher.

Approach: binarised image patch matching via Normalised Cross-Correlation (NCC)
with optional SSIM refinement at the best position.

Scoring
-------
Raw NCC peak values are unreliable as a standalone confidence measure — a chart
with no resemblance to the pattern will still produce *some* best-window score.
Instead we use a **heatmap z-score**: how many standard deviations the best NCC
value sits above the mean of the entire heatmap.  A genuine match produces a
sharp, isolated peak (high z-score); a spurious best-window on a non-matching
chart produces a flat or noisy heatmap (z-score near 0).

The final reported similarity is a blend of:
  • normalised z-score  — "does a clear peak exist anywhere?"  (primary)
  • spatial clustering  — "do the top-k hits agree on a location?"

SSIM was removed: binarised chart images are ~97% black background, so SSIM
is dominated by shared empty space and scores high on unrelated windows.

Template scaling
----------------
The reference patch is scaled to the target width (time-stretch invariance)
but its HEIGHT IS SCALED PROPORTIONALLY — it is NOT stretched to fill the
full candidate height.  The proportionally-scaled template is then centred
vertically in a same-height canvas.

This preserves relative vertical position, which is the primary signal that
distinguishes a real match from an inverted or unrelated pattern on actual
chart data.  Full-height fill collapsed all vertical information and made
the matcher unable to distinguish a rising pattern from a falling one.

Invariances
-----------
* Time stretch             : reference patch tried at several horizontal scales
* Chart theme              : both images binarised before comparison
* Magnitude (partial)      : proportional scaling keeps relative price position;
                             patterns at very different absolute price levels that
                             happen to occupy the same vertical chart fraction will
                             still match — this is usually the desired behaviour.

API
---
Public entry point: ``visual_search_chart``
Returns (similarity: float, start_pct: float, end_pct: float) — same contract
as ``search_chart`` in engine.py.

Optional dependencies
---------------------
* scikit-image : no longer used for scoring. Import retained for potential
                 future use but the matcher runs identically with or without it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np

# ── Optional skimage ──────────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as _ssim_fn
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False


# ─────────────────────────────────────────────────────────────────────
#  CONSTANTS  (mirror engine.py crop defaults)
# ─────────────────────────────────────────────────────────────────────

_TOP_CROP_PCT    = 0.04
_BOTTOM_CROP_PCT = 0.08
_RIGHT_CROP_PCT  = 0.09

_N_SCALES    = 9
_MIN_PATCH_W = 20
_BINARISE_THRESH = 60

# z-score cap before normalising to [0,1].
# Real matches typically z > 5; spurious z < 3.
_ZSCORE_CAP = 10.0

# Top-k windows used for the spatial-clustering check.
_CLUSTER_K = 5

# Final score blend weights.
_W_ZSCORE  = 0.75
_W_CLUSTER = 0.25


# ─────────────────────────────────────────────────────────────────────
#  PUBLIC PARAMS DATACLASS
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VisualMatchParams:
    """
    Tuning knobs for the visual matcher.

    n_scales        : number of horizontal scale candidates (more = better
                      time-stretch coverage, slower).  Default 9.

    scale_min       : narrowest reference patch relative to its original width.
    scale_max       : widest reference patch relative to its original width.

    binarise_thresh : brightness threshold for stripping background/grid.
                      Lower catches dimmer chart lines; higher is more selective.

    stride_pct      : sliding-window stride as a fraction of reference patch width.
                      0.05 = 5% step (precise); 0.20 = 20% step (fast).

    min_window_pct  : minimum window width as a fraction of candidate chart width.
                      Windows narrower than this are skipped regardless of scale.
                      Prevents tiny compressed templates from producing spurious
                      high-NCC hits against a few lucky columns.  Default 0.05
                      (5% of chart width).
    """
    n_scales:        int   = _N_SCALES
    scale_min:       float = 0.5
    scale_max:       float = 2.0
    binarise_thresh: int   = _BINARISE_THRESH
    stride_pct:      float = 0.10
    min_window_pct:  float = 0.05


# ─────────────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────

def _crop_chart_region(img_bgr: np.ndarray) -> np.ndarray:
    """Crop axis/label strips; return grayscale chart area."""
    h, w = img_bgr.shape[:2]
    tc = int(h * _TOP_CROP_PCT)
    bc = int(h * _BOTTOM_CROP_PCT)
    rc = int(w * _RIGHT_CROP_PCT)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray[tc: h - bc, 0: w - rc]


def _binarise(gray: np.ndarray, thresh: int) -> np.ndarray:
    """Return uint8 binary image: price pixels → 255, background → 0."""
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


def _make_template(
    ref_bin: np.ndarray,
    target_w: int,
    canvas_h: int,
) -> np.ndarray:
    """
    Scale ref_bin to target_w wide with proportional height, then centre
    it vertically in a canvas_h-tall zero canvas.

    This preserves the relative vertical position of the price pattern,
    which is the main signal distinguishing a genuine match from an
    inverted or unrelated pattern on real chart data.
    """
    ref_h, ref_w = ref_bin.shape
    new_h = max(1, int(ref_h * (target_w / ref_w)))
    new_h = min(new_h, canvas_h)

    resized = cv2.resize(ref_bin, (target_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((canvas_h, target_w), dtype=np.uint8)
    v_offset = (canvas_h - new_h) // 2
    v_offset = max(0, min(v_offset, canvas_h - new_h))
    canvas[v_offset: v_offset + new_h, :] = resized
    return canvas


def _ncc_heatmap(template_f32: np.ndarray, cand_f32: np.ndarray) -> np.ndarray:
    """
    Run cv2.matchTemplate (TM_CCOEFF_NORMED).
    Template must have the same height as cand, so result shape is (1, W).
    Returns a flattened 1-D float32 array.
    """
    result = cv2.matchTemplate(cand_f32, template_f32, cv2.TM_CCOEFF_NORMED)
    return result.ravel()


def _zscore_similarity(heatmap: np.ndarray) -> tuple[float, int]:
    """
    How many std-devs does the best position stand above the heatmap mean?
    Returns (normalised_score_in_[0,1], best_position_index).
    """
    if heatmap.size == 0:
        return 0.0, 0

    peak_idx = int(np.argmax(heatmap))
    peak_val = float(heatmap[peak_idx])
    std      = float(np.std(heatmap))

    if std < 1e-8:
        return 0.0, peak_idx

    z = (peak_val - float(np.mean(heatmap))) / std
    return float(np.clip(z / _ZSCORE_CAP, 0.0, 1.0)), peak_idx


def _cluster_score(heatmap: np.ndarray, best_idx: int, patch_w: int) -> float:
    """
    Fraction of the top-k peaks that cluster within one patch-width of the best.
    1.0 = tight cluster (genuine match), ~0 = scattered peaks (noise).
    """
    k = min(_CLUSTER_K, len(heatmap))
    if k < 2:
        return 1.0
    top_k_idx  = np.argpartition(heatmap, -k)[-k:]
    neighbours = np.sum(np.abs(top_k_idx - best_idx) <= patch_w)
    return float(np.clip((neighbours - 1) / (k - 1), 0.0, 1.0))


def _ssim_at_position(
    template_u8: np.ndarray,
    cand_bin: np.ndarray,
    start: int,
) -> float:
    """
    SSIM between template and the candidate window at `start`, computed only
    over the rows that contain price content in the template.

    Computing SSIM over the full canvas is misleading: binarised chart images
    are mostly black background, so two unrelated windows both score high SSIM
    simply because they share large empty regions.  Restricting to the non-zero
    row band of the template focuses the comparison on actual price structure.

    Returns score in [0, 1], or 0 if skimage is unavailable or template is blank.
    """
    if not _SKIMAGE_OK:
        return 0.0
    end = start + template_u8.shape[1]
    region = cand_bin[:, start:end]
    if region.shape != template_u8.shape:
        return 0.0

    # Find rows in the template that contain any price pixels
    active_rows = np.where(template_u8.any(axis=1))[0]
    if len(active_rows) < 2:
        return 0.0

    r0, r1 = int(active_rows[0]), int(active_rows[-1]) + 1
    tmpl_crop   = template_u8[r0:r1, :].astype(np.float32)
    region_crop = region[r0:r1, :].astype(np.float32)

    # SSIM needs window_size <= min dimension; skip if crop is too thin
    min_dim = min(tmpl_crop.shape)
    if min_dim < 3:
        return 0.0
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score, _ = _ssim_fn(
            tmpl_crop,
            region_crop,
            full=True,
            data_range=255,
            win_size=win_size,
        )
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────
#  REFERENCE PATCH PREPARATION
# ─────────────────────────────────────────────────────────────────────

def prepare_visual_reference(
    ref_img_bgr: np.ndarray,
    params: VisualMatchParams | None = None,
    apply_crop: bool = False,
) -> np.ndarray | None:
    """
    Extract and binarise the chart area from the reference image.

    apply_crop=False (default): the image is used as-is.  This matches
    engine.extract_pattern's behaviour for references — the expected workflow
    is a tight crop of just the pattern, with no axis labels to remove.
    Set apply_crop=True if the reference is a full chart screenshot with
    axis/label areas.

    Returns a 2-D uint8 binary image, or None if extraction fails.
    """
    if params is None:
        params = VisualMatchParams()
    if apply_crop:
        gray = _crop_chart_region(ref_img_bgr)
    else:
        gray = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2GRAY)
    binary = _binarise(gray, params.binarise_thresh)
    if binary.sum() < 500:
        return None
    return binary


# ─────────────────────────────────────────────────────────────────────
#  MAIN SEARCH
# ─────────────────────────────────────────────────────────────────────

def visual_search_chart(
    ref_patch: np.ndarray,
    cand_img_bgr: np.ndarray,
    params: VisualMatchParams | None = None,
) -> tuple[float, float, float]:
    """
    Slide the reference patch across the candidate chart image and return
    the best visual match.

    Parameters
    ----------
    ref_patch     : binarised reference patch from ``prepare_visual_reference``
    cand_img_bgr  : raw BGR candidate chart image
    params        : tuning knobs (uses defaults if None)

    Returns
    -------
    (similarity, start_pct, end_pct)  — all floats, similarity in [0, 1]
    """
    if params is None:
        params = VisualMatchParams()

    cand_gray = _crop_chart_region(cand_img_bgr)
    cand_bin  = _binarise(cand_gray, params.binarise_thresh)
    cand_h, cand_w = cand_bin.shape
    cand_f32  = cand_bin.astype(np.float32)

    ref_h, ref_w = ref_patch.shape

    scales = np.exp(
        np.linspace(np.log(params.scale_min), np.log(params.scale_max), params.n_scales)
    )

    # Minimum window width in pixels — prevents tiny compressed templates from
    # producing spurious high-NCC hits against a few lucky columns.
    min_win_px = max(_MIN_PATCH_W, int(cand_w * params.min_window_pct))

    # Best record across all scales: (z_sim, cluster, ssim, start_idx, patch_w)
    # Scale is selected by highest z-score, not raw NCC peak — this penalises
    # small windows that achieve high NCC by coincidence on a handful of columns.
    best_z_sim  = -np.inf
    best_record = (0.0, 0.0, 0, ref_w)

    for scale in scales:
        scaled_w = int(ref_w * scale)
        if scaled_w < min_win_px or scaled_w > cand_w:
            continue

        # Proportional height scaling — preserves vertical position
        template_u8  = _make_template(ref_patch, scaled_w, cand_h)
        template_f32 = template_u8.astype(np.float32)

        heatmap = _ncc_heatmap(template_f32, cand_f32)

        z_sim, peak_idx = _zscore_similarity(heatmap)

        # Select the scale whose heatmap peak stands out most clearly above
        # background — high z-score means a genuine localised match, not just
        # a lucky high-NCC window at a compressed scale.
        if z_sim > best_z_sim:
            best_z_sim = z_sim
            cluster    = _cluster_score(heatmap, peak_idx, scaled_w)
            best_record = (z_sim, cluster, peak_idx, scaled_w)

    z_sim, cluster, best_start, best_pw = best_record

    # ── Blend into final score ────────────────────────────────────────
    similarity = _W_ZSCORE * z_sim + _W_CLUSTER * cluster

    best_end  = best_start + best_pw
    start_pct = best_start / max(cand_w, 1)
    end_pct   = min(best_end / max(cand_w, 1), 1.0)

    return float(similarity), float(start_pct), float(end_pct)


# ─────────────────────────────────────────────────────────────────────
#  TARGETED SCORING — verify a specific window (hybrid mode)
# ─────────────────────────────────────────────────────────────────────

def visual_score_at(
    ref_patch: np.ndarray,
    cand_img_bgr: np.ndarray,
    start_pct: float,
    end_pct: float,
    params: VisualMatchParams | None = None,
) -> float:
    """
    Score how visually similar a SPECIFIC window of the candidate chart is
    to the reference patch.  Used by the hybrid engine to verify the window
    that DTW selected, rather than searching independently.

    The score is the z-score of the NCC value at the given position relative
    to the full heatmap at that template width — i.e. "does this particular
    window stand out as a match compared to everywhere else on this chart?"

    Returns a float in [0, 1].
    """
    if params is None:
        params = VisualMatchParams()

    cand_gray = _crop_chart_region(cand_img_bgr)
    cand_bin  = _binarise(cand_gray, params.binarise_thresh)
    cand_h, cand_w = cand_bin.shape
    cand_f32  = cand_bin.astype(np.float32)

    start_px = int(np.clip(start_pct, 0.0, 1.0) * cand_w)
    end_px   = int(np.clip(end_pct,   0.0, 1.0) * cand_w)
    win_w    = end_px - start_px
    if win_w < _MIN_PATCH_W:
        return 0.0

    # DTW's window boundaries are approximate — its pixel positions come from
    # 1-D signal space and won't align exactly with the visual optimum.
    # Try a few width variants and allow positional tolerance proportional to
    # the window size; report the best.
    best = 0.0
    for w_factor in (0.85, 1.0, 1.15):
        t_w = int(win_w * w_factor)
        if t_w < _MIN_PATCH_W or t_w > cand_w:
            continue

        template_u8  = _make_template(ref_patch, t_w, cand_h)
        template_f32 = template_u8.astype(np.float32)

        heatmap = _ncc_heatmap(template_f32, cand_f32)
        if heatmap.size == 0:
            continue
        std = float(np.std(heatmap))
        if std < 1e-8:
            continue

        # Positional tolerance: ±20% of window width around the DTW start
        tol = max(1, int(win_w * 0.20))
        lo  = max(0, start_px - tol)
        hi  = min(len(heatmap), start_px + tol + 1)
        if lo >= hi:
            continue
        val_at = float(heatmap[lo:hi].max())

        z = (val_at - float(np.mean(heatmap))) / std
        best = max(best, float(np.clip(z / _ZSCORE_CAP, 0.0, 1.0)))

    return best


# ─────────────────────────────────────────────────────────────────────
#  CONVENIENCE: batch search
# ─────────────────────────────────────────────────────────────────────

def visual_search_batch(
    ref_img_bgr: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
    params: VisualMatchParams | None = None,
    progress_cb=None,
) -> list[tuple[str, float, float, float]]:
    """
    Search a list of candidate images against a reference.

    Parameters
    ----------
    ref_img_bgr  : raw BGR reference image
    candidates   : list of (filename, bgr_image) tuples
    params       : VisualMatchParams (uses defaults if None)
    progress_cb  : optional callable(i, total, filename) for progress updates

    Returns
    -------
    List of (filename, similarity, start_pct, end_pct), unsorted.
    """
    if params is None:
        params = VisualMatchParams()
    ref_patch = prepare_visual_reference(ref_img_bgr, params)
    if ref_patch is None:
        return []
    results = []
    for i, (name, img_bgr) in enumerate(candidates):
        if progress_cb:
            progress_cb(i, len(candidates), name)
        sim, s_pct, e_pct = visual_search_chart(ref_patch, img_bgr, params)
        results.append((name, sim, s_pct, e_pct))
    return results
