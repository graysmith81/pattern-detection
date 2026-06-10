"""
engine.py — Pattern extraction and matching core.
Shared between the console app and the Streamlit UI.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

DTW_RESOLUTION = 100   # fixed length for DTW comparison


# ─────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    chart_path: str
    similarity: float          # 0–1
    match_start_x: int         # pixel column in the extracted curve
    match_end_x: int
    curve_len: int             # total width of extracted curve
    start_time: str = ""       # OCR'd timestamp at match start
    end_time: str   = ""       # OCR'd timestamp at match end
    symbol: str     = ""       # parsed from filename if available
    timeframe: str  = ""       # parsed from filename if available
    dtw_similarity: float    = 0.0   # hybrid mode: DTW shape sub-score
    visual_similarity: float = 0.0   # hybrid mode: visual texture sub-score

    @property
    def start_pct(self) -> float:
        return self.match_start_x / max(self.curve_len, 1)

    @property
    def end_pct(self) -> float:
        return self.match_end_x / max(self.curve_len, 1)


@dataclass
class ChartSignals:
    """
    Raw pixel-space signals extracted from a chart image.

    All arrays are smoothed but NOT globally normalised — normalisation
    happens per-window inside search_chart so local amplitude is preserved.

    Line / HLC chart:
      mid    : (H+L)/2 per column — price direction / shape
      spread : L-H per column     — H-L band width (volatility proxy)

    Candlestick chart (all of the above, plus):
      body_mid    : (body_top + body_bot) / 2 — open/close midpoint
      body_spread : body_bot - body_top        — body height (conviction)
      direction   : +1 bull (close > open), -1 bear, 0 doji/unknown
    """
    mid:          np.ndarray
    spread:       np.ndarray
    body_mid:     np.ndarray | None = None
    body_spread:  np.ndarray | None = None
    direction:    np.ndarray | None = None   # float array: +1 / -1 / 0

    @property
    def length(self) -> int:
        return len(self.mid)

    @property
    def has_candle_signals(self) -> bool:
        return self.body_mid is not None


@dataclass
class CandleColors:
    """
    HSV colour ranges for candlestick body detection.
    Defaults match TradingView dark theme (green/red candles, white wicks).
    Override for other platforms or themes.
    """
    bull_lower: tuple = (45, 80, 60)    # green bodies
    bull_upper: tuple = (85, 255, 255)
    bear_lower1: tuple = (0,  150, 80)  # red bodies (hue wraps at 180)
    bear_upper1: tuple = (15, 255, 255)
    bear_lower2: tuple = (155, 150, 80)
    bear_upper2: tuple = (180, 255, 255)
    wick_lower: tuple = (0, 0, 180)     # white/grey wicks
    wick_upper: tuple = (180, 20, 255)


# Singleton default — import and use directly, or override per-call
TRADINGVIEW_COLORS = CandleColors()


class SignalMode:
    MID_ONLY = "mid"     # match wick mid curve only — works for any chart type
    HLC      = "hlc"     # match wick mid + wick spread — HLC / line charts
    CANDLE   = "candle"  # match wick mid + body mid + body spread + direction


class ReferenceMode:
    DRAWN_LINE  = "drawn"   # image has a coloured markup line drawn on it
    PLAIN_CHART = "plain"   # image is a clean line/HLC/candle chart — no markup


# ─────────────────────────────────────────────────────────────────────
#  EXTRACTION — REFERENCE (drawn line)
# ─────────────────────────────────────────────────────────────────────

def extract_drawn_line(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Extract a hand-drawn markup line from a reference chart image.
    Detects blue (primary) or red (fallback) overlay lines.
    Returns a normalised 1-D float array in [0, 1], or None on failure.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Blue
    mask = cv2.inRange(hsv, np.array([90, 80, 80]), np.array([140, 255, 255]))

    # Red fallback
    if mask.sum() < 1000:
        m1 = cv2.inRange(hsv, np.array([0, 100, 100]),   np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        mask = m1 | m2

    if mask.sum() < 500:
        return None

    h, w = mask.shape
    y_vals = np.full(w, np.nan)
    for x in range(w):
        col = np.where(mask[:, x] > 0)[0]
        if len(col):
            y_vals[x] = float(np.median(col))

    y_vals = _interp_nans(y_vals)
    if y_vals is None:
        return None

    valid = np.where(~np.isnan(y_vals))[0]
    if len(valid) < 10:
        return None
    y_vals = y_vals[valid[0]: valid[-1] + 1]

    if len(y_vals) > 11:
        y_vals = savgol_filter(y_vals, window_length=11, polyorder=2)

    return _norm_flip(y_vals)


# ─────────────────────────────────────────────────────────────────────
#  UNIFIED PATTERN EXTRACTION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def extract_pattern(
    img_bgr: np.ndarray,
    mode: str = ReferenceMode.PLAIN_CHART,
    signal_mode: str = SignalMode.MID_ONLY,
    brightness_threshold: int = 150,
    candle_colors: CandleColors | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> ChartSignals | None:
    """
    Unified entry point for extracting a pattern signature from a reference image.

    mode=DRAWN_LINE : detects a coloured (blue/red) markup overlay
    mode=PLAIN_CHART: treats the image itself as the pattern
    signal_mode     : MID_ONLY | HLC | CANDLE
    candle_colors   : override default TradingView colours (CANDLE mode only)
    crop            : optional pixel crop (x0, y0, x1, y1) applied first

    Returns a ChartSignals with signals normalised to [0,1].
    """
    if crop is not None:
        x0, y0, x1, y1 = crop
        img_bgr = img_bgr[y0:y1, x0:x1]

    if mode == ReferenceMode.DRAWN_LINE:
        raw_mid = extract_drawn_line(img_bgr)
        if raw_mid is None:
            return None
        return ChartSignals(mid=raw_mid, spread=np.zeros_like(raw_mid))

    elif signal_mode == SignalMode.CANDLE:
        signals = extract_candlestick(
            img_bgr,
            colors=candle_colors,
            top_crop_pct=0.0,
            bottom_crop_pct=0.0,
            right_crop_pct=0.0,
        )
        if signals is None:
            return None
        return _normalise_signals(signals, signal_mode)

    else:
        signals = extract_price_curve(
            img_bgr,
            top_crop_pct=0.0,
            bottom_crop_pct=0.0,
            right_crop_pct=0.0,
            brightness_threshold=brightness_threshold,
        )
        if signals is None:
            return None
        return _normalise_signals(signals, signal_mode)


# ─────────────────────────────────────────────────────────────────────
#  EXTRACTION — CANDIDATE (high/low line chart)
# ─────────────────────────────────────────────────────────────────────

def extract_price_curve(
    img_bgr: np.ndarray,
    dark_bg: bool = True,
    top_crop_pct: float = 0.04,
    bottom_crop_pct: float = 0.08,
    right_crop_pct: float = 0.09,
    brightness_threshold: int = 180,
) -> ChartSignals | None:
    """
    Extract price signals from a chart image.

    For a single-line chart  → mid is the close line, spread is ~0
    For an HLC / band chart  → mid is (H+L)/2, spread captures the H-L width

    Returns a ChartSignals object with raw (un-normalised) pixel values,
    or None if extraction fails.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    tc = int(h * top_crop_pct)
    bc = int(h * bottom_crop_pct)
    rc = int(w * right_crop_pct)
    chart = gray[tc: h - bc, 0: w - rc]
    ch, cw = chart.shape

    _, bright = cv2.threshold(chart, brightness_threshold, 255, cv2.THRESH_BINARY)

    high_y  = np.full(cw, np.nan)   # topmost bright pixel  (= highest price)
    low_y   = np.full(cw, np.nan)   # bottommost bright pixel (= lowest price)

    for x in range(cw):
        col = np.where(bright[:, x] > 0)[0]
        if len(col) >= 2:
            high_y[x] = float(col.min())   # low pixel Y → high on screen → high price
            low_y[x]  = float(col.max())
        elif len(col) == 1:
            high_y[x] = low_y[x] = float(col[0])

    mid_y    = (high_y + low_y) / 2.0
    spread_y = low_y - high_y            # always ≥ 0; larger = wider H-L band

    mid_y    = _interp_nans(mid_y)
    spread_y = _interp_nans(spread_y)
    if mid_y is None:
        return None
    if spread_y is None:
        spread_y = np.zeros_like(mid_y)

    win = min(21, max(5, (len(mid_y) // 5) * 2 + 1))
    if len(mid_y) > win:
        mid_y    = savgol_filter(mid_y,    window_length=win, polyorder=2)
        spread_y = savgol_filter(spread_y, window_length=win, polyorder=2)

    return ChartSignals(mid=mid_y, spread=spread_y)


def extract_candlestick(
    img_bgr: np.ndarray,
    colors: CandleColors | None = None,
    top_crop_pct: float = 0.04,
    bottom_crop_pct: float = 0.08,
    right_crop_pct: float = 0.09,
) -> ChartSignals | None:
    """
    Extract price signals from a TradingView-style candlestick chart.

    Signals returned (all raw pixel values, un-normalised):
      mid         : (wick_high + wick_low) / 2  — overall price level
      spread      : wick_low - wick_high         — wick-to-wick range (H-L)
      body_mid    : (body_top + body_bot) / 2   — open/close midpoint
      body_spread : body_bot - body_top          — body height (conviction)
      direction   : +1 bull, -1 bear, 0 doji/gap — candle colour per column
    """
    if colors is None:
        colors = TRADINGVIEW_COLORS

    h, w = img_bgr.shape[:2]
    tc = int(h * top_crop_pct)
    bc = int(h * bottom_crop_pct)
    rc = int(w * right_crop_pct)

    chart_bgr = img_bgr[tc: h - bc, 0: w - rc]
    chart_hsv = cv2.cvtColor(chart_bgr, cv2.COLOR_BGR2HSV)
    chart_gray = cv2.cvtColor(chart_bgr, cv2.COLOR_BGR2GRAY)
    ch, cw = chart_bgr.shape[:2]

    # ── Colour masks ──────────────────────────────────────────────────
    bull_mask = cv2.inRange(chart_hsv,
                            np.array(colors.bull_lower),
                            np.array(colors.bull_upper))

    bear_mask = (cv2.inRange(chart_hsv,
                             np.array(colors.bear_lower1),
                             np.array(colors.bear_upper1)) |
                 cv2.inRange(chart_hsv,
                             np.array(colors.bear_lower2),
                             np.array(colors.bear_upper2)))

    wick_mask = cv2.inRange(chart_hsv,
                            np.array(colors.wick_lower),
                            np.array(colors.wick_upper))

    # Any bright pixel = part of a candle (body or wick)
    _, candle_mask = cv2.threshold(chart_gray, 60, 255, cv2.THRESH_BINARY)

    # ── Per-column extraction ─────────────────────────────────────────
    wick_high  = np.full(cw, np.nan)   # topmost candle pixel  (= price high)
    wick_low   = np.full(cw, np.nan)   # bottommost candle pixel (= price low)
    body_top   = np.full(cw, np.nan)   # topmost body pixel    (open or close)
    body_bot   = np.full(cw, np.nan)   # bottommost body pixel (open or close)
    direction  = np.zeros(cw)          # +1 bull, -1 bear

    for x in range(cw):
        # Full candle extent (wick tips)
        all_rows = np.where(candle_mask[:, x] > 0)[0]
        if len(all_rows) >= 1:
            wick_high[x] = float(all_rows.min())
            wick_low[x]  = float(all_rows.max())

        # Bull body
        g_rows = np.where(bull_mask[:, x] > 0)[0]
        if len(g_rows) >= 2:
            body_top[x]  = float(g_rows.min())
            body_bot[x]  = float(g_rows.max())
            direction[x] = +1.0
            continue

        # Bear body
        r_rows = np.where(bear_mask[:, x] > 0)[0]
        if len(r_rows) >= 2:
            body_top[x]  = float(r_rows.min())
            body_bot[x]  = float(r_rows.max())
            direction[x] = -1.0

    # ── Interpolate gaps ──────────────────────────────────────────────
    wick_high = _interp_nans(wick_high)
    wick_low  = _interp_nans(wick_low)
    body_top  = _interp_nans(body_top)
    body_bot  = _interp_nans(body_bot)

    if wick_high is None or wick_low is None:
        return None
    if body_top is None:
        body_top = (wick_high + wick_low) / 2
    if body_bot is None:
        body_bot = body_top.copy()

    # ── Smooth ────────────────────────────────────────────────────────
    win = min(21, max(5, (cw // 5) * 2 + 1))
    wick_high  = savgol_filter(wick_high,  win, 2)
    wick_low   = savgol_filter(wick_low,   win, 2)
    body_top   = savgol_filter(body_top,   win, 2)
    body_bot   = savgol_filter(body_bot,   win, 2)

    mid_y        = (wick_high + wick_low) / 2.0
    spread_y     = wick_low  - wick_high          # ≥ 0
    body_mid_y   = (body_top + body_bot) / 2.0
    body_spread_y = body_bot - body_top            # ≥ 0

    return ChartSignals(
        mid=mid_y,
        spread=spread_y,
        body_mid=body_mid_y,
        body_spread=body_spread_y,
        direction=direction,
    )

def extract_timestamps(
    img_bgr: np.ndarray,
    bottom_crop_pct: float = 0.06,
    right_crop_pct: float = 0.07,
) -> list[tuple[float, str]]:
    """
    OCR the time labels along the bottom X-axis.
    Returns a list of (x_fraction, label_string) tuples, sorted by x.
    Returns empty list silently if tesseract is not installed.
    """
    try:
        import pytesseract
    except ImportError:
        return []

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        # tesseract binary not found on this system
        return []

    h, w = img_bgr.shape[:2]
    bc = int(h * bottom_crop_pct)
    rc = int(w * right_crop_pct)

    # Crop just the bottom axis strip
    axis_strip = img_bgr[h - bc: h, 0: w - rc]

    # Invert if dark background so OCR sees black text on white
    gray = cv2.cvtColor(axis_strip, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # OCR with bounding boxes
    try:
        data = pytesseract.image_to_data(
            binary,
            config="--psm 11 -c tessedit_char_whitelist=0123456789:.",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []

    results = []
    strip_w = axis_strip.shape[1]
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if re.match(r"^\d{1,2}:\d{2}$", text) or re.match(r"^\d{4}-\d{2}-\d{2}$", text):
            x_centre = data["left"][i] + data["width"][i] / 2
            results.append((x_centre / strip_w, text))

    return sorted(results, key=lambda t: t[0])


def interpolate_time(
    x_fraction: float,
    timestamps: list[tuple[float, str]],
) -> str:
    """Given an x_fraction (0–1), interpolate the approximate timestamp."""
    if not timestamps:
        return f"{x_fraction:.0%}"
    if len(timestamps) == 1:
        return timestamps[0][1]

    # Find surrounding pair
    for i in range(len(timestamps) - 1):
        x0, t0 = timestamps[i]
        x1, t1 = timestamps[i + 1]
        if x0 <= x_fraction <= x1:
            # Try to interpolate if times are HH:MM
            try:
                def to_mins(s):
                    h, m = s.split(":")
                    return int(h) * 60 + int(m)
                def from_mins(m):
                    m = int(m) % (24 * 60)
                    return f"{m // 60:02d}:{m % 60:02d}"
                frac = (x_fraction - x0) / max(x1 - x0, 1e-9)
                interp = to_mins(t0) + frac * (to_mins(t1) - to_mins(t0))
                return from_mins(interp)
            except Exception:
                return t0
    # Outside range — clamp
    return timestamps[0][1] if x_fraction < timestamps[0][0] else timestamps[-1][1]


# ─────────────────────────────────────────────────────────────────────
#  FUZZINESS PARAMETERS
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FuzzParams:
    """
    Controls how strictly the shape must match.

    dtw_resolution  : number of points both curves are resampled to before
                      DTW comparison. Lower = coarser comparison = fuzzier.
                      Range: 20 (very fuzzy) – 150 (very strict).

    dtw_band_pct    : Sakoe-Chiba warp band as a fraction of sequence length.
                      Higher = more time-stretch tolerance.
                      Range: 0.10 – 0.60.

    score_decay     : exponent in exp(-dist * decay). Lower = gentler score
                      falloff so near-matches still score respectably.
                      Range: 4 (gentle) – 15 (strict).

    smooth_window   : Savitzky-Golay window applied to candidate windows
                      before comparison (must be odd, >= 3). Higher removes
                      more noise.  Range: 3 – 31.
    """
    dtw_resolution: int   = 100
    dtw_band_pct:   float = 0.20
    score_decay:    float = 10.0
    smooth_window:  int   = 5

    def __post_init__(self):
        # Ensure smooth_window is odd
        if self.smooth_window % 2 == 0:
            self.smooth_window += 1


# Master-slider preset builder
def fuzz_preset(level: float) -> FuzzParams:
    """
    Build a FuzzParams from a single fuzziness level in [0.0, 1.0].

    0.0 = strict  (original behaviour)
    0.5 = balanced
    1.0 = very fuzzy (catches visually similar but not pixel-perfect matches)
    """
    # Each parameter interpolates linearly between its strict and fuzzy endpoints
    def lerp(a, b, t):
        return a + (b - a) * t

    return FuzzParams(
        dtw_resolution = int(lerp(100, 25,  level)),   # 100 → 25
        dtw_band_pct   =     lerp(0.20, 0.50, level),  # 20% → 50%
        score_decay    =     lerp(10.0, 5.0,  level),  # 10  → 5
        smooth_window  = int(lerp(5,    29,   level)) | 1,  # 5 → 29 (kept odd)
    )


# ─────────────────────────────────────────────────────────────────────
#  DTW
# ─────────────────────────────────────────────────────────────────────

def dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    band: int | None = None,
    fuzz: FuzzParams | None = None,
) -> float:
    """
    Banded (Sakoe-Chiba) DTW distance, normalised by path length.

    Row-vectorised: the cost row and the 2-way min of the previous row are
    computed with numpy; only the left-neighbour scan remains a Python loop
    (it has a sequential dependency that can't vectorise).
    """
    n, m = len(a), len(b)
    if band is None:
        band_pct = fuzz.dtw_band_pct if fuzz else 0.20
        band = max(1, int(band_pct * max(n, m)))

    INF = np.inf
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    curr = np.full(m + 1, INF)

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        curr.fill(INF)

        # Vectorised: cost row and min(diag, up) for the band
        cost = np.abs(a[i - 1] - b[j_lo - 1: j_hi])          # len = j_hi-j_lo+1
        best2 = np.minimum(prev[j_lo: j_hi + 1],              # up
                           prev[j_lo - 1: j_hi])              # diagonal

        # Sequential left-neighbour scan (cannot vectorise)
        left = curr[j_lo - 1]
        for k in range(j_hi - j_lo + 1):
            v = cost[k] + (best2[k] if best2[k] < left else left)
            curr[j_lo + k] = v
            left = v

        prev, curr = curr, prev

    return float(prev[m]) / (n + m)


def similarity_score(dist: float, fuzz: FuzzParams | None = None) -> float:
    decay = fuzz.score_decay if fuzz else 10.0
    return float(np.exp(-dist * decay))


def _velocity_profile(curve_ds: np.ndarray) -> np.ndarray:
    """
    Rate-of-change profile of a normalised, resampled curve, mapped to [0,1].

    This is the signal that distinguishes a sharp move from a slow drift:
    after window-local price normalisation, a 3-day drift and a 6-hour crash
    can have identical price *shapes*, but their velocity profiles differ —
    the crash concentrates movement into a spike, the drift spreads it flat.

    Velocity is normalised by its own max magnitude so the profile captures
    *where* the movement happens within the window, not absolute speed
    (which is meaningless across different image resolutions).
    """
    v = np.gradient(curve_ds)
    m = float(np.max(np.abs(v)))
    if m < 1e-9:
        return np.full_like(curve_ds, 0.5)
    return (v / m + 1.0) / 2.0


# ─────────────────────────────────────────────────────────────────────
#  SLIDING WINDOW SEARCH
# ─────────────────────────────────────────────────────────────────────

def search_chart(
    pattern: ChartSignals,
    signals: ChartSignals,
    stride: int = 5,
    min_window_ratio: float = 0.5,
    max_window_ratio: float = 2.0,
    context_ratio: float = 3.0,
    signal_mode: str = SignalMode.MID_ONLY,
    spread_weight: float = 0.35,
    velocity_weight: float = 0.30,
    fuzz: FuzzParams | None = None,
) -> tuple[float, int, int]:
    """
    Slide windows across `signals` and return the best match against `pattern`.

    signal_mode=MID_ONLY : match wick mid only
    signal_mode=HLC      : match wick mid (65%) + wick spread (35%)
    signal_mode=CANDLE   : match wick mid (40%) + body mid (40%) + body spread (20%)

    velocity_weight controls how much the rate-of-change profile contributes
    to the score (0 = disabled, pure shape matching). Velocity matching
    distinguishes sharp moves from slow drifts with the same net shape —
    DTW's time-warping otherwise treats them as identical.

    fuzz controls matching strictness — see FuzzParams / fuzz_preset().
    Returns (similarity, start_x, end_x).
    """
    if fuzz is None:
        fuzz = FuzzParams()

    use_candle = (signal_mode == SignalMode.CANDLE)
    use_spread = (signal_mode == SignalMode.HLC)
    res        = fuzz.dtw_resolution

    # ── Pre-compute pattern fingerprints ──────────────────────────────
    pat_mid_ds = _resample(_smooth(pattern.mid, fuzz.smooth_window), res)
    pat_vel_ds = _velocity_profile(pat_mid_ds)

    if use_candle and pattern.has_candle_signals:
        pat_body_mid_ds    = _resample(_smooth(pattern.body_mid,    fuzz.smooth_window), res)
        pat_body_spread_ds = _resample(_smooth(pattern.body_spread, fuzz.smooth_window), res)
        # weights: wick mid 40%, body mid 40%, body spread 20%
        w_mid = 0.40; w_body_mid = 0.40; w_body_sp = 0.20
    elif use_spread:
        pat_spread_ds = _resample(_smooth(pattern.spread, fuzz.smooth_window), res)
        w_mid = 1.0 - spread_weight; w_body_mid = 0.0; w_body_sp = 0.0
    else:
        w_mid = 1.0; w_body_mid = 0.0; w_body_sp = 0.0

    pat_len   = len(pattern.mid)
    curve_len = signals.length

    best_score, best_start, best_end = 0.0, 0, min(pat_len, curve_len)

    min_win   = max(10, int(pat_len * min_window_ratio))
    max_win   = min(curve_len, int(pat_len * max_window_ratio))
    win_sizes = np.linspace(min_win, max_win, 8, dtype=int)

    def _local_norm_flip(arr):
        """Local normalise + flip Y."""
        s = _smooth(arr, fuzz.smooth_window)
        rng = max(s.max() - s.min(), 1e-6)
        return 1.0 - (s - s.min()) / rng

    def _local_norm(arr):
        """Local normalise, no flip (for spread / body height)."""
        s = _smooth(arr, fuzz.smooth_window)
        rng = max(s.max() - s.min(), 1e-6)
        return (s - s.min()) / rng

    def _sim(a_ds, b_arr, flip=True):
        b_n  = _local_norm_flip(b_arr) if flip else _local_norm(b_arr)
        dist = dtw_distance(a_ds, _resample(b_n, res), fuzz=fuzz)
        return similarity_score(dist, fuzz=fuzz)

    for win_len in win_sizes:
        ctx_half = int(win_len * context_ratio / 2)

        for start in range(0, curve_len - win_len + 1, stride):
            seg_mid   = signals.mid[start: start + win_len]
            mid_range = seg_mid.max() - seg_mid.min()
            if mid_range < 1e-6:
                continue

            # Resample once — reused for both shape and velocity matching
            seg_mid_ds = _resample(_local_norm_flip(seg_mid), res)
            mid_sim = similarity_score(
                dtw_distance(pat_mid_ds, seg_mid_ds, fuzz=fuzz), fuzz=fuzz)

            # ── Prune: best possible combined score for this window ──
            # Assume every not-yet-computed component scores a perfect 1.0;
            # if even that can't beat the current best, skip the expensive
            # remaining DTW computations.
            optimistic_shape = w_mid * mid_sim + (1.0 - w_mid)
            optimistic = 0.80 * ((1.0 - velocity_weight) * optimistic_shape
                                 + velocity_weight) + 0.20
            if optimistic <= best_score:
                continue

            # Velocity profile match — distinguishes sharp moves from drifts
            if velocity_weight > 0:
                vel_sim = similarity_score(
                    dtw_distance(pat_vel_ds, _velocity_profile(seg_mid_ds),
                                 fuzz=fuzz), fuzz=fuzz)
            else:
                vel_sim = 0.0

            if use_candle and signals.has_candle_signals:
                body_mid_sim = _sim(pat_body_mid_ds,
                                    signals.body_mid[start: start + win_len], flip=True)
                body_sp_sim  = _sim(pat_body_spread_ds,
                                    signals.body_spread[start: start + win_len], flip=False)
                shape_sim = (w_mid * mid_sim
                             + w_body_mid * body_mid_sim
                             + w_body_sp  * body_sp_sim)

            elif use_spread:
                sp_sim    = _sim(pat_spread_ds,
                                 signals.spread[start: start + win_len], flip=False)
                shape_sim = w_mid * mid_sim + spread_weight * sp_sim

            else:
                shape_sim = mid_sim

            # Blend velocity into the shape score
            if velocity_weight > 0:
                shape_sim = (1.0 - velocity_weight) * shape_sim \
                            + velocity_weight * vel_sim

            # Amplitude tiebreaker (context-relative)
            ctx_s     = max(0, start - ctx_half)
            ctx_e     = min(curve_len, start + win_len + ctx_half)
            ctx_range = signals.mid[ctx_s:ctx_e].max() - signals.mid[ctx_s:ctx_e].min()
            amp_ratio = mid_range / ctx_range if ctx_range > 1e-6 else 0.0

            combined = 0.80 * shape_sim + 0.20 * amp_ratio
            if combined > best_score:
                best_score, best_start, best_end = combined, start, start + win_len

    # ── Recalculate final reported similarity at best position ────────
    seg_best_ds = _resample(_local_norm_flip(signals.mid[best_start:best_end]), res)
    best_sim = similarity_score(
        dtw_distance(pat_mid_ds, seg_best_ds, fuzz=fuzz), fuzz=fuzz) * w_mid

    if use_candle and signals.has_candle_signals:
        best_sim += (w_body_mid * _sim(pat_body_mid_ds,
                                       signals.body_mid[best_start:best_end], flip=True)
                   + w_body_sp  * _sim(pat_body_spread_ds,
                                       signals.body_spread[best_start:best_end], flip=False))
    elif use_spread:
        best_sim += spread_weight * _sim(pat_spread_ds,
                                         signals.spread[best_start:best_end], flip=False)

    if velocity_weight > 0:
        best_vel_sim = similarity_score(
            dtw_distance(pat_vel_ds, _velocity_profile(seg_best_ds), fuzz=fuzz),
            fuzz=fuzz)
        best_sim = (1.0 - velocity_weight) * best_sim \
                   + velocity_weight * best_vel_sim

    return best_sim, best_start, best_end



def score_window(
    pattern: ChartSignals,
    signals: ChartSignals,
    start: int,
    end: int,
    signal_mode: str = SignalMode.MID_ONLY,
    spread_weight: float = 0.35,
    velocity_weight: float = 0.30,
    fuzz: FuzzParams | None = None,
) -> float:
    """
    DTW-score a SPECIFIC window of `signals` against `pattern`, without
    searching.  Used by the hybrid engine to cross-score a window proposed
    by the visual engine.  Mirrors the scoring in search_chart.

    Returns similarity in [0, 1]; 0.0 if the window is degenerate.
    """
    if fuzz is None:
        fuzz = FuzzParams()

    start = max(0, int(start))
    end   = min(signals.length, int(end))
    if end - start < 10:
        return 0.0

    seg_mid = signals.mid[start:end]
    if seg_mid.max() - seg_mid.min() < 1e-6:
        return 0.0

    res = fuzz.dtw_resolution
    pat_mid_ds = _resample(_smooth(pattern.mid, fuzz.smooth_window), res)
    pat_vel_ds = _velocity_profile(pat_mid_ds)

    def _norm_flip_local(arr):
        s = _smooth(arr, fuzz.smooth_window)
        rng = max(s.max() - s.min(), 1e-6)
        return 1.0 - (s - s.min()) / rng

    def _norm_local(arr):
        s = _smooth(arr, fuzz.smooth_window)
        rng = max(s.max() - s.min(), 1e-6)
        return (s - s.min()) / rng

    seg_mid_ds = _resample(_norm_flip_local(seg_mid), res)
    sim = similarity_score(
        dtw_distance(pat_mid_ds, seg_mid_ds, fuzz=fuzz), fuzz=fuzz)

    use_candle = (signal_mode == SignalMode.CANDLE)
    use_spread = (signal_mode == SignalMode.HLC)

    if use_candle and pattern.has_candle_signals and signals.has_candle_signals:
        pat_bm_ds = _resample(_smooth(pattern.body_mid,    fuzz.smooth_window), res)
        pat_bs_ds = _resample(_smooth(pattern.body_spread, fuzz.smooth_window), res)
        bm = _resample(_norm_flip_local(signals.body_mid[start:end]), res)
        bs = _resample(_norm_local(signals.body_spread[start:end]), res)
        bm_sim = similarity_score(dtw_distance(pat_bm_ds, bm, fuzz=fuzz), fuzz=fuzz)
        bs_sim = similarity_score(dtw_distance(pat_bs_ds, bs, fuzz=fuzz), fuzz=fuzz)
        sim = 0.40 * sim + 0.40 * bm_sim + 0.20 * bs_sim
    elif use_spread:
        pat_sp_ds = _resample(_smooth(pattern.spread, fuzz.smooth_window), res)
        sp = _resample(_norm_local(signals.spread[start:end]), res)
        sp_sim = similarity_score(dtw_distance(pat_sp_ds, sp, fuzz=fuzz), fuzz=fuzz)
        sim = (1.0 - spread_weight) * sim + spread_weight * sp_sim

    if velocity_weight > 0:
        vel_sim = similarity_score(
            dtw_distance(pat_vel_ds, _velocity_profile(seg_mid_ds), fuzz=fuzz),
            fuzz=fuzz)
        sim = (1.0 - velocity_weight) * sim + velocity_weight * vel_sim

    return float(sim)


# ─────────────────────────────────────────────────────────────────────
#  FILENAME METADATA PARSING
# ─────────────────────────────────────────────────────────────────────
def parse_filename_metadata(path: str) -> dict:
    """
    Attempt to parse symbol, timeframe, and datetime from a filename.
    Supports patterns like:
      AAPL_1m_2024-01-15_09-30.png
      NQ_5m_20240115.png
      chart_EURUSD_H1_2024-01-15T09:30.png
    """
    stem = Path(path).stem
    meta = {"symbol": "", "timeframe": "", "datetime": ""}

    # Timeframe
    tf = re.search(r"\b(\d+[mMhHdDwW]|M\d+|H\d+|D\d+)\b", stem)
    if tf:
        meta["timeframe"] = tf.group(1)

    # Date
    dt = re.search(r"(\d{4}[-_]\d{2}[-_]\d{2})", stem)
    if dt:
        meta["datetime"] = dt.group(1).replace("_", "-")

    # Time component
    tm = re.search(r"(\d{2}[-:]\d{2})(?:[-:]\d{2})?$", stem)
    if tm and meta["datetime"]:
        meta["datetime"] += " " + tm.group(1).replace("-", ":")

    # Symbol — heuristic: uppercase word that isn't the timeframe or date
    parts = re.split(r"[-_]", stem)
    for p in parts:
        if re.match(r"^[A-Z]{2,6}$", p) and p != meta["timeframe"].upper():
            meta["symbol"] = p
            break

    return meta


# ─────────────────────────────────────────────────────────────────────
#  MATCH HIGHLIGHT — draw overlay on image
# ─────────────────────────────────────────────────────────────────────

def draw_match_highlight(
    img_bgr: np.ndarray,
    start_pct: float,
    end_pct: float,
    right_crop_pct: float = 0.09,
    color: tuple = (0, 255, 100),
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Draw a semi-transparent highlight over the matched region.
    Returns a new BGR image.

    right_crop_pct must match the crop used during extraction (0.09),
    otherwise the highlight is horizontally offset from the true match.
    """
    h, w = img_bgr.shape[:2]
    rc = int(w * right_crop_pct)
    chart_w = w - rc

    x0 = int(start_pct * chart_w)
    x1 = int(end_pct   * chart_w)

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, 0), (x1, h), color, thickness=-1)
    result = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

    # Border line
    cv2.rectangle(result, (x0, 0), (x1, h), color, thickness=2)
    return result


def _normalise_signals(signals: ChartSignals, signal_mode: str) -> ChartSignals:
    """
    Normalise a raw ChartSignals (as returned by extract_*) to [0,1] for use
    as a reference pattern. Candidate signals are normalised per-window inside
    search_chart instead.
    """
    mid_n = _norm_flip(signals.mid)
    if mid_n is None:
        mid_n = np.zeros(signals.length)

    def norm_positive(arr):
        """Normalise a ≥0 array (spread, body height) to [0,1]."""
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 1e-6 else np.zeros_like(arr)

    if signal_mode == SignalMode.HLC:
        spread_n = norm_positive(signals.spread)
        return ChartSignals(mid=mid_n, spread=spread_n)

    elif signal_mode == SignalMode.CANDLE:
        spread_n = norm_positive(signals.spread)
        body_mid_n = _norm_flip(signals.body_mid) if signals.body_mid is not None \
                     else mid_n.copy()
        body_spread_n = norm_positive(signals.body_spread) if signals.body_spread is not None \
                        else np.zeros_like(mid_n)
        dir_n = signals.direction if signals.direction is not None \
                else np.zeros_like(mid_n)
        return ChartSignals(
            mid=mid_n,
            spread=spread_n,
            body_mid=body_mid_n,
            body_spread=body_spread_n,
            direction=dir_n,
        )

    else:  # MID_ONLY
        return ChartSignals(mid=mid_n, spread=np.zeros_like(mid_n))


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply Savitzky-Golay smoothing. Window is clamped to array length and kept odd."""
    window = max(3, min(window, len(arr) - (1 if len(arr) % 2 == 0 else 0)))
    if window % 2 == 0:
        window -= 1
    if len(arr) <= window:
        return arr.copy()
    return savgol_filter(arr, window_length=window, polyorder=2)


def _interp_nans(arr: np.ndarray) -> np.ndarray | None:
    nans = np.isnan(arr)
    if nans.all():
        return None
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def _norm_flip(arr: np.ndarray) -> np.ndarray | None:
    arr = arr - arr.min()
    rng = arr.max()
    if rng < 1e-6:
        return None
    return 1.0 - (arr / rng)


def _resample(arr: np.ndarray, n: int) -> np.ndarray:
    src = np.linspace(0, 1, len(arr))
    dst = np.linspace(0, 1, n)
    return np.interp(dst, src, arr)


def load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    return img


def image_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
