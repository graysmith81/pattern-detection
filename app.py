"""
app.py — Chart Pattern Matcher — Streamlit UI
----------------------------------------------
Run with:
    streamlit run app.py

Two matching engines are available via tabs:
  • DTW (Dynamic Time Warping) — original pixel-signal sliding-window search
  • CNN Embedding             — zero-shot ResNet/EfficientNet cosine retrieval
                               (only shown when torch + torchvision are installed)
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from engine import (
    extract_pattern,
    extract_price_curve,
    extract_candlestick,
    extract_timestamps,
    interpolate_time,
    search_chart,
    draw_match_highlight,
    parse_filename_metadata,
    image_from_bytes,
    MatchResult,
    ReferenceMode,
    SignalMode,
    FuzzParams,
    fuzz_preset,
    CandleColors,
    TRADINGVIEW_COLORS,
)

# CNN engine — optional; degrades gracefully if PyTorch is absent
try:
    from embedder import (
        TORCH_AVAILABLE,
        load_model,
        embed_image,
        build_index,
        save_index,
        load_index,
        query_index,
        EmbedResult,
        ChartIndex,
    )
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chart Pattern Matcher",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .time-badge {
        background: #2d3250;
        border-radius: 6px;
        padding: 4px 10px;
        font-family: monospace;
        font-size: 0.9rem;
        color: #90caf9;
    }
    .meta-tag {
        background: #37404e;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        color: #ccc;
        margin-right: 4px;
    }
    .mode-pill {
        display: inline-block;
        background: #2d3250;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        color: #90caf9;
        margin-bottom: 8px;
    }
    .signal-hlc { color: #ffd54f !important; }
    .signal-mid { color: #90caf9 !important; }
    .cnn-badge {
        display: inline-block;
        background: #1b3a2d;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        color: #69f0ae;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    # ── DTW settings ──────────────────────────────────────────────────
    st.subheader("DTW — Reference mode")
    ref_mode_label = st.radio(
        "How is the pattern defined?",
        options=["Plain line chart", "Drawn markup line"],
        index=0,
        help=(
            "Plain: upload a clean line chart — the line itself is the pattern.\n"
            "Drawn: upload a chart with a blue or red annotation drawn on it."
        ),
    )
    ref_mode = (
        ReferenceMode.PLAIN_CHART
        if ref_mode_label == "Plain line chart"
        else ReferenceMode.DRAWN_LINE
    )

    st.subheader("DTW — Signal mode")
    st.caption(
        "Choose which signals to extract and match. "
        "Both images must use the same chart type to unlock richer matching."
    )

    CHART_TYPES = ["Single line (mid only)", "HLC (mid + spread)", "Candlestick (green/red)"]

    ref_signal_label = st.radio(
        "Reference chart type", CHART_TYPES, index=0, key="ref_sig",
        help="What type of chart is your reference image?"
    )
    cand_signal_label = st.radio(
        "Candidate chart type", CHART_TYPES, index=0, key="cand_sig",
        help="What type of charts are your candidate images?"
    )

    def _label_to_signal(label):
        if "Candlestick" in label: return SignalMode.CANDLE
        if "HLC"         in label: return SignalMode.HLC
        return SignalMode.MID_ONLY

    ref_signal  = _label_to_signal(ref_signal_label)
    cand_signal = _label_to_signal(cand_signal_label)

    if ref_signal == cand_signal == SignalMode.CANDLE:
        match_signal  = SignalMode.CANDLE
        signal_status = "🕯️ Candlestick matching (wick mid + body mid + body spread)"
    elif ref_signal == cand_signal == SignalMode.HLC:
        match_signal  = SignalMode.HLC
        signal_status = "🟡 HLC matching (mid + spread)"
    else:
        match_signal  = SignalMode.MID_ONLY
        if ref_signal != cand_signal:
            signal_status = "🔵 Mid-only matching (chart types differ — using wick mid only)"
        else:
            signal_status = "🔵 Mid-only matching"
    st.info(signal_status)

    if match_signal == SignalMode.HLC:
        spread_weight = st.slider(
            "Spread weight in HLC matching", 0.10, 0.60, 0.35, 0.05,
            help="How much the H-L spread signal contributes vs mid-price shape",
        )
    else:
        spread_weight = 0.35

    st.subheader("DTW — Search")
    threshold = st.slider("Minimum similarity (DTW)", 0.20, 0.95, 0.40, 0.05)
    top_n     = st.slider("Max results", 1, 20, 8)
    stride    = st.select_slider(
        "Search precision",
        options=[2, 5, 10, 20, 40],
        value=10,
        help="Lower = more precise but slower",
    )

    st.subheader("🎚️ Fuzziness")
    st.caption(
        "Higher fuzziness catches visually similar but not pixel-perfect matches. "
        "Lower fuzziness is more discriminating."
    )
    fuzz_level = st.slider(
        "Fuzziness", 0.0, 1.0, 0.25, 0.05,
        help="Master control — adjusts DTW resolution, warp band, score decay, and smoothing together"
    )

    with st.expander("⚙️ Advanced fuzziness controls", expanded=False):
        st.caption("Override individual parameters. Overrides the master slider.")
        use_advanced = st.toggle("Use advanced settings", value=False)
        if use_advanced:
            adv_resolution = st.slider(
                "DTW resolution (points)", 20, 150, 100,
                help="Lower = coarser comparison = fuzzier shape matching"
            )
            adv_band = st.slider(
                "Warp band %", 10, 60, 20,
                help="How much DTW can stretch/compress time. Higher = more time-shift tolerance"
            )
            adv_decay = st.slider(
                "Score decay", 4.0, 15.0, 10.0, 0.5,
                help="Lower = gentler score dropoff for near-matches"
            )
            adv_smooth = st.slider(
                "Smoothing window (px)", 3, 31, 5, 2,
                help="Larger window removes more noise before comparison"
            )
            fuzz_params = FuzzParams(
                dtw_resolution=adv_resolution,
                dtw_band_pct=adv_band / 100,
                score_decay=adv_decay,
                smooth_window=adv_smooth | 1,   # keep odd
            )
        else:
            fuzz_params = fuzz_preset(fuzz_level)
            st.caption(
                f"→ Resolution: **{fuzz_params.dtw_resolution}** pts  |  "
                f"Warp: **{fuzz_params.dtw_band_pct:.0%}**  |  "
                f"Decay: **{fuzz_params.score_decay:.1f}**  |  "
                f"Smooth: **{fuzz_params.smooth_window}** px"
            )

    st.subheader("Chart visuals")
    brightness_thresh = st.slider(
        "Line brightness threshold", 80, 240, 150,
        help="Brightness cutoff for the price line",
    )

    # ── CNN settings ──────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        st.subheader("CNN — Embedding")
        cnn_backbone = st.selectbox(
            "Backbone",
            options=["resnet50", "efficientnet_b0"],
            index=0,
            help="resnet50 → 2048-dim · efficientnet_b0 → 1280-dim",
        )
        cnn_threshold = st.slider(
            "Minimum similarity (CNN)", 0.00, 1.00, 0.50, 0.01,
            help="Cosine similarity cutoff for CNN results",
        )

        st.subheader("CNN — Index")
        index_path = st.text_input(
            "Index file path",
            value="chart_index.npz",
            help="Embeddings are saved/loaded here so charts are not re-embedded each session",
        )
        col_save, col_clear = st.columns(2)
        with col_save:
            save_index_btn = st.button("💾 Save index", use_container_width=True,
                                       help="Save the current in-memory index to disk")
        with col_clear:
            clear_index_btn = st.button("🗑️ Clear index", use_container_width=True,
                                        help="Remove in-memory index (disk file is kept)")
    else:
        cnn_backbone   = "resnet50"
        cnn_threshold  = 0.50
        index_path     = "chart_index.npz"
        save_index_btn = False
        clear_index_btn = False

    st.subheader("TradingView MCP")
    st.info(
        "Connect a TradingView MCP server to auto-populate candidate charts.\n\n"
        "Filename format: `SYMBOL_TF_YYYY-MM-DD_HH-MM.png`",
        icon="🔌",
    )

    st.divider()
    st.caption("Chart Pattern Matcher v1.3")


# ─────────────────────────────────────────────────────────────────────
#  SESSION STATE — CNN index
# ─────────────────────────────────────────────────────────────────────

if "cnn_index" not in st.session_state:
    st.session_state.cnn_index = None    # ChartIndex | None

# Handle sidebar index actions
if TORCH_AVAILABLE:
    if clear_index_btn:
        st.session_state.cnn_index = None
        st.sidebar.success("In-memory index cleared.")

    if save_index_btn:
        if st.session_state.cnn_index is not None and st.session_state.cnn_index.size > 0:
            save_index(st.session_state.cnn_index, index_path)
            st.sidebar.success(f"Index saved → {index_path}")
        else:
            st.sidebar.warning("Nothing to save — run CNN search first to build the index.")


# ─────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────

st.title("📈 Chart Pattern Matcher")
st.caption("Upload a reference pattern → drop candidate charts → find visual matches")


# ─────────────────────────────────────────────────────────────────────
#  REFERENCE UPLOAD  (shared by both engines)
# ─────────────────────────────────────────────────────────────────────

col_ref, col_cands = st.columns([1, 1], gap="large")

with col_ref:
    st.subheader("① Reference pattern")

    if ref_mode == ReferenceMode.PLAIN_CHART:
        if ref_signal == SignalMode.CANDLE:
            st.caption("Upload a **candlestick chart** (TradingView default colours). Wick + body signals will be extracted.")
            mode_hint = "🕯️ Candlestick chart"
        elif ref_signal == SignalMode.HLC:
            st.caption("Upload a clean **HLC line chart**. Mid-price + H-L spread will be extracted.")
            mode_hint = "🟡 HLC plain chart"
        else:
            st.caption("Upload a clean **single-line chart**. The price line defines the pattern.")
            mode_hint = "🔵 Single-line plain chart"
    else:
        st.caption("Upload a chart with a **blue or red line drawn** on it.")
        mode_hint = "✏️ Drawn markup"

    st.markdown(f'<div class="mode-pill">{mode_hint}</div>', unsafe_allow_html=True)

    ref_file = st.file_uploader(
        "Drop reference image here",
        type=["png", "jpg", "jpeg", "bmp"],
        key="ref",
        label_visibility="collapsed",
    )

    pattern     = None
    ref_img_bgr = None

    if ref_file:
        ref_img_bgr = image_from_bytes(ref_file.read())
        h, w = ref_img_bgr.shape[:2]

        st.image(
            cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption=ref_file.name,
        )

        crop = None
        if ref_mode == ReferenceMode.PLAIN_CHART:
            with st.expander("✂️ Crop reference image (optional)", expanded=False):
                st.caption("Trim axis labels or whitespace before extraction.")
                cc1, cc2 = st.columns(2)
                with cc1:
                    crop_left   = st.number_input("Left px",   0, w - 1, 0)
                    crop_top    = st.number_input("Top px",    0, h - 1, 0)
                with cc2:
                    crop_right  = st.number_input("Right px",  1, w, w)
                    crop_bottom = st.number_input("Bottom px", 1, h, h)

                if (crop_left, crop_top, crop_right, crop_bottom) != (0, 0, w, h):
                    crop = (int(crop_left), int(crop_top), int(crop_right), int(crop_bottom))
                    preview = ref_img_bgr[crop[1]:crop[3], crop[0]:crop[2]]
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                             caption="Cropped region", use_container_width=True)

        with st.spinner("Extracting pattern…"):
            pattern = extract_pattern(
                ref_img_bgr,
                mode=ref_mode,
                signal_mode=ref_signal,
                brightness_threshold=brightness_thresh,
                candle_colors=TRADINGVIEW_COLORS,
                crop=crop,
            )

        if pattern is not None:
            st.success(f"✓ Pattern extracted — {pattern.length} sample points")

            chart_data = {"Mid (price shape)": pattern.mid.tolist()}
            if ref_signal == SignalMode.HLC and pattern.spread.max() > 0.01:
                chart_data["Spread (H-L width)"] = pattern.spread.tolist()
                st.caption("🟡 Showing mid + spread signals")
            else:
                st.caption("🔵 Showing mid signal")
            st.line_chart(chart_data, height=140)
        else:
            st.error(
                "Could not extract a pattern. "
                + ("Ensure a blue/red line is drawn on the chart."
                   if ref_mode == ReferenceMode.DRAWN_LINE
                   else "Try lowering the brightness threshold in the sidebar.")
            )
    else:
        st.info("👆 Drop your reference chart here")


# ─────────────────────────────────────────────────────────────────────
#  CANDIDATE CHARTS  (shared by both engines)
# ─────────────────────────────────────────────────────────────────────

with col_cands:
    st.subheader("② Candidate charts")

    if cand_signal == SignalMode.HLC:
        st.caption(
            "Upload **HLC line charts** to search. "
            "Mid-price + spread will be extracted from each."
        )
    else:
        st.caption(
            "Upload chart images to search. "
            "Filenames like `NQ_1m_2026-05-08_04-16.png` will be parsed for metadata."
        )

    cand_files = st.file_uploader(
        "Drop candidate charts here",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        key="cands",
        label_visibility="collapsed",
    )

    if cand_files:
        st.success(f"✓ {len(cand_files)} image(s) loaded")
        thumb_cols = st.columns(min(len(cand_files), 4))
        for i, f in enumerate(cand_files[:4]):
            with thumb_cols[i]:
                img = image_from_bytes(f.read())
                f.seek(0)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         use_container_width=True, caption=Path(f.name).stem[:18])
        if len(cand_files) > 4:
            st.caption(f"…and {len(cand_files) - 4} more")
    else:
        st.info("👆 Drop your candidate charts here")


# ─────────────────────────────────────────────────────────────────────
#  SHARED RESULT RENDERER  (DTW)
#  Defined before the tabs so it is in scope when called inside Tab 1.
# ─────────────────────────────────────────────────────────────────────

def _render_dtw_results(results: list, cand_files: list) -> None:
    """Render DTW MatchResult cards.  Identical to the original app.py output."""
    for rank, result in enumerate(results, 1):
        sim_pct = result.similarity
        colour  = "#00e676" if sim_pct >= 0.6 else "#ff9800"

        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            tags = ""
            if result.symbol:
                tags += f'<span class="meta-tag">{result.symbol}</span>'
            if result.timeframe:
                tags += f'<span class="meta-tag">{result.timeframe}</span>'
            st.markdown(
                f"**#{rank} — {Path(result.chart_path).stem}** &nbsp; {tags}",
                unsafe_allow_html=True,
            )
        with hcol2:
            st.markdown(
                f'<div style="text-align:right;font-size:1.5rem;'
                f'font-weight:bold;color:{colour}">{sim_pct:.1%}</div>',
                unsafe_allow_html=True,
            )

        st.progress(sim_pct)

        if result.start_time and result.end_time:
            time_str = (
                f'<span class="time-badge">⏱ {result.start_time} → {result.end_time}</span>'
            )
        else:
            time_str = (
                f'<span class="time-badge">'
                f'📍 {result.start_pct:.0%} – {result.end_pct:.0%} of chart width'
                f'</span>'
            )
        st.markdown(time_str, unsafe_allow_html=True)

        match_file = next((f for f in cand_files if f.name == result.chart_path), None)
        if match_file:
            match_file.seek(0)
            match_img   = image_from_bytes(match_file.read())
            highlighted = draw_match_highlight(match_img, result.start_pct, result.end_pct)
            st.image(
                cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="Match region highlighted in green",
            )
            _, buf = cv2.imencode(".png", highlighted)
            st.download_button(
                label="⬇ Download highlighted image",
                data=buf.tobytes(),
                file_name=f"match_{rank}_{result.chart_path}",
                mime="image/png",
                key=f"dtw_dl_{rank}",
            )

        st.divider()


# ─────────────────────────────────────────────────────────────────────
#  ENGINE TABS
# ─────────────────────────────────────────────────────────────────────

st.divider()

tab_labels = ["🔵 DTW matching"]
if TORCH_AVAILABLE:
    tab_labels.append("🟢 CNN embedding")

tabs = st.tabs(tab_labels)


# ═════════════════════════════════════════════════════════════════════
#  TAB 1 — DTW  (completely unchanged logic)
# ═════════════════════════════════════════════════════════════════════

with tabs[0]:

    if pattern is not None and cand_files:
        if match_signal == SignalMode.HLC:
            st.info(
                f"🟡 **HLC matching** — comparing mid-price shape "
                f"({100-int(spread_weight*100)}%) + H-L spread ({int(spread_weight*100)}%)",
                icon="ℹ️",
            )
        else:
            note = ""
            if ref_signal != cand_signal:
                note = " (HLC spread ignored — both images must be HLC to use spread matching)"
            st.info(f"🔵 **Mid-only matching** — shape comparison only{note}", icon="ℹ️")

    run_disabled = (pattern is None) or (not cand_files)

    if st.button(
        "🔍 Search for pattern",
        disabled=run_disabled,
        type="primary",
        use_container_width=True,
        key="dtw_run",
    ):
        st.subheader("③ Results")
        st.caption(
            f"Fuzziness: **{fuzz_level:.0%}** — "
            f"Resolution {fuzz_params.dtw_resolution}pt · "
            f"Warp {fuzz_params.dtw_band_pct:.0%} · "
            f"Decay {fuzz_params.score_decay:.1f} · "
            f"Smooth {fuzz_params.smooth_window}px"
        )

        results: list[MatchResult] = []
        prog = st.progress(0, text="Searching…")

        for i, f in enumerate(cand_files):
            f.seek(0)
            img_bgr = image_from_bytes(f.read())
            prog.progress(i / len(cand_files), text=f"Searching {f.name}…")

            if cand_signal == SignalMode.CANDLE:
                signals = extract_candlestick(img_bgr)
            else:
                signals = extract_price_curve(img_bgr, brightness_threshold=brightness_thresh)
            if signals is None:
                continue

            sim, sx, ex = search_chart(
                pattern, signals,
                stride=stride,
                signal_mode=match_signal,
                spread_weight=spread_weight,
                fuzz=fuzz_params,
            )

            timestamps = extract_timestamps(img_bgr)
            start_time = interpolate_time(sx / max(signals.length, 1), timestamps)
            end_time   = interpolate_time(ex / max(signals.length, 1), timestamps)
            meta       = parse_filename_metadata(f.name)

            results.append(MatchResult(
                chart_path=f.name,
                similarity=sim,
                match_start_x=sx,
                match_end_x=ex,
                curve_len=signals.length,
                start_time=start_time,
                end_time=end_time,
                symbol=meta["symbol"],
                timeframe=meta["timeframe"],
            ))

        prog.progress(1.0, text="Done")

        results = sorted(
            [r for r in results if r.similarity >= threshold],
            key=lambda r: r.similarity,
            reverse=True,
        )[:top_n]

        if not results:
            st.warning(
                f"No matches found above {threshold:.0%} similarity. "
                "Try lowering the threshold in the sidebar."
            )
        else:
            st.success(f"Found {len(results)} match(es)")
            _render_dtw_results(results, cand_files)

    elif run_disabled:
        if pattern is None and not cand_files:
            st.info("Upload a reference image and candidate charts to begin.")
        elif pattern is None:
            st.info("Upload a reference image to begin.")
        else:
            st.info("Upload candidate charts to search.")


# ═════════════════════════════════════════════════════════════════════
#  TAB 2 — CNN  (only present when TORCH_AVAILABLE)
# ═════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    with tabs[1]:

        st.markdown(
            '<div class="cnn-badge">🟢 Zero-shot CNN embedding · cosine similarity retrieval</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Backbone: **{cnn_backbone}** · "
            "Invariant to scale, magnitude, and time-stretch by design "
            "(all crops are resized to a fixed 224×224 canvas before embedding)."
        )

        # ── Index status ──────────────────────────────────────────────
        idx: ChartIndex | None = st.session_state.cnn_index

        col_idx_a, col_idx_b = st.columns([3, 1])
        with col_idx_a:
            if idx is not None and idx.size > 0:
                st.success(
                    f"✓ Index ready — **{idx.size}** chart(s) embedded "
                    f"with `{idx.backbone}`"
                )
            else:
                st.info("No index in memory. Run a search to build one, or load from disk.")
        with col_idx_b:
            if st.button("📂 Load index from disk", use_container_width=True, key="cnn_load"):
                try:
                    loaded = load_index(index_path)
                    st.session_state.cnn_index = loaded
                    st.success(f"Loaded {loaded.size} embeddings from {index_path}")
                    st.rerun()
                except FileNotFoundError:
                    st.error(f"File not found: {index_path}")

        # ── Search button ─────────────────────────────────────────────
        cnn_disabled = (ref_img_bgr is None) or (not cand_files)

        if st.button(
            "🔍 Search with CNN",
            disabled=cnn_disabled,
            type="primary",
            use_container_width=True,
            key="cnn_run",
        ):
            # 1. Load / reuse backbone
            with st.spinner(f"Loading {cnn_backbone} (first run downloads weights)…"):
                try:
                    model = load_model(cnn_backbone)
                except Exception as e:
                    st.error(f"Failed to load backbone: {e}")
                    st.stop()

            # 2. Embed reference image
            with st.spinner("Embedding reference chart…"):
                try:
                    query_vec = embed_image(ref_img_bgr, model)
                except Exception as e:
                    st.error(f"Reference embedding failed: {e}")
                    st.stop()

            # 3. Build / update index from candidate files
            #    If the existing index was built with the same backbone and
            #    already contains all currently uploaded filenames, reuse it.
            existing: ChartIndex | None = st.session_state.cnn_index
            uploaded_names = {f.name for f in cand_files}

            need_rebuild = (
                existing is None
                or existing.size == 0
                or existing.backbone != cnn_backbone
                or not uploaded_names.issubset(set(existing.filenames))
            )

            if need_rebuild:
                # Collect (name, bytes) pairs — seek to 0 first
                image_pairs = []
                for f in cand_files:
                    f.seek(0)
                    image_pairs.append((f.name, f.read()))

                prog = st.progress(0, text="Building embedding index…")

                def _cb(i, total, name):
                    prog.progress(
                        i / max(total, 1),
                        text=f"Embedding {Path(name).stem[:30]}… ({i+1}/{total})"
                    )

                with st.spinner("Embedding candidate charts…"):
                    new_index = build_index(image_pairs, model,
                                            backbone=cnn_backbone,
                                            progress_callback=_cb)

                prog.progress(1.0, text="Done")
                st.session_state.cnn_index = new_index
                idx = new_index
            else:
                idx = existing
                st.caption("♻️ Reusing existing index (same backbone, same files).")

            if idx.size == 0:
                st.warning("No charts could be embedded. Check the uploaded images.")
                st.stop()

            # 4. Query
            results_cnn = query_index(
                idx,
                query_vec,
                top_n=top_n,
                threshold=cnn_threshold,
            )

            # Restore file seek positions for display
            for f in cand_files:
                f.seek(0)

            # 5. Render results
            st.subheader("③ Results")
            st.caption(
                f"Cosine similarity threshold: **{cnn_threshold:.0%}** · "
                f"Showing top **{top_n}**"
            )

            if not results_cnn:
                st.warning(
                    f"No matches found above {cnn_threshold:.0%} cosine similarity. "
                    "Try lowering the threshold in the sidebar."
                )
            else:
                st.success(f"Found {len(results_cnn)} match(es)")

                for rank, result in enumerate(results_cnn, 1):
                    sim_pct = result.similarity
                    colour  = "#00e676" if sim_pct >= 0.70 else "#ff9800"

                    hcol1, hcol2 = st.columns([3, 1])
                    with hcol1:
                        tags = ""
                        if result.symbol:
                            tags += f'<span class="meta-tag">{result.symbol}</span>'
                        if result.timeframe:
                            tags += f'<span class="meta-tag">{result.timeframe}</span>'
                        if result.datetime:
                            tags += f'<span class="meta-tag">{result.datetime}</span>'
                        st.markdown(
                            f"**#{rank} — {Path(result.chart_path).stem}** &nbsp; {tags}",
                            unsafe_allow_html=True,
                        )
                    with hcol2:
                        st.markdown(
                            f'<div style="text-align:right;font-size:1.5rem;'
                            f'font-weight:bold;color:{colour}">{sim_pct:.1%}</div>',
                            unsafe_allow_html=True,
                        )

                    st.progress(sim_pct)

                    # CNN match is whole-image — highlight the full chart width
                    match_file = next(
                        (f for f in cand_files if f.name == result.chart_path), None
                    )
                    if match_file:
                        match_file.seek(0)
                        match_img   = image_from_bytes(match_file.read())
                        # Full-width highlight (0 % → 100 %)
                        highlighted = draw_match_highlight(
                            match_img, start_pct=0.0, end_pct=1.0,
                            color=(100, 255, 160), alpha=0.15,
                        )
                        st.image(
                            cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                            caption=f"Cosine similarity: {sim_pct:.3f}",
                        )
                        _, buf = cv2.imencode(".png", highlighted)
                        st.download_button(
                            label="⬇ Download highlighted image",
                            data=buf.tobytes(),
                            file_name=f"cnn_match_{rank}_{result.chart_path}",
                            mime="image/png",
                            key=f"cnn_dl_{rank}",
                        )

                    st.divider()

        elif cnn_disabled:
            if ref_img_bgr is None and not cand_files:
                st.info("Upload a reference image and candidate charts to begin.")
            elif ref_img_bgr is None:
                st.info("Upload a reference image to begin.")
            else:
                st.info("Upload candidate charts to search.")



