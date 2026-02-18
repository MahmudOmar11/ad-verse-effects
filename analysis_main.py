#!/usr/bin/env python3
"""
Ad-verse Effects v5 -- Analysis & Visualization
=================================================
Handles multiple models, system prompt conditions, and
provides AGGREGATED + STRATIFIED output.

PRIMARY QUESTION: How much does each ad shift the model toward
the advertised product?

SECONDARY QUESTIONS:
  - Does the system prompt persona affect susceptibility?
  - Do smaller models differ from flagship models?
  - How do wellness supplements differ from Rx drugs?

Accepts one or more pipeline output files (Excel or JSONL).

Usage:
    python ad_verse_analysis_v5.py [file1.xlsx] [file2.xlsx] ...
    python ad_verse_analysis_v5.py results/              # all .xlsx in dir
"""

import sys, math, glob
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

try:
    from scipy import stats as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────
BG      = "#F8F9FB"
WHITE   = "#FFFFFF"
GRID    = "#E5E9F0"
TXT     = "#1E2A3A"
SUB     = "#7C879A"
GREEN   = "#0EA47A"
RED     = "#E5484D"
BLUE    = "#3E82FC"
AMBER   = "#F5A623"
INDIGO  = "#6C5CE7"
SLATE   = "#A0AEC0"
TEAL    = "#14B8A6"
CORAL   = "#FF6B6B"
PURPLE  = "#A855F7"

# Prompt condition colors
PROMPT_COLORS = {
    "physician":        BLUE,
    "helpful_ai":       AMBER,
    "customer_service":  RED,
    "no_persona":       SLATE,
}

PROMPT_LABELS = {
    "physician":         "Physician",
    "helpful_ai":        "Helpful AI",
    "customer_service":  "Customer Service",
    "no_persona":        "No Persona",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": WHITE,
        "axes.edgecolor": GRID, "axes.labelcolor": TXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "text.color": TXT,
        "xtick.color": SUB, "ytick.color": SUB,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.labelsize": 11, "axes.titlesize": 13,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7,
    })


# ─────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────
def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0, 0, 0
    p = k / n
    z = sp.norm.ppf(1 - alpha / 2) if HAS_SCIPY else 1.96
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d
    return p, max(0, c - m), min(1, c + m)


def cohens_h(p1, p2):
    return 2 * (math.asin(math.sqrt(max(0, min(1, p1)))) -
                math.asin(math.sqrt(max(0, min(1, p2)))))


# ─────────────────────────────────────────────────────────────
# LOAD -- handles multiple files
# ─────────────────────────────────────────────────────────────
def load_one(path: Path) -> pd.DataFrame:
    """Load a single pipeline output file."""
    if path.suffix == ".jsonl":
        rows = []
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        return pd.DataFrame(rows)
    else:
        g = pd.read_excel(path, sheet_name="Grading")
        g.columns = g.columns.str.strip()
        return g


def load_all(paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate multiple output files."""
    frames = []
    for p in paths:
        df = load_one(p)
        # Ensure standard columns
        for col in ["is_correct", "chose_advertised", "chose_competitor", "parse_ok"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("boolean")
        # Backfill system_prompt_name if missing (v4 files)
        if "system_prompt_name" not in df.columns:
            df["system_prompt_name"] = "physician"
        if "model" not in df.columns:
            # Try to infer from filename
            name = p.stem
            model_str = name.replace("adverse_", "").rsplit("_", 2)[0]
            df["model"] = model_str
        if "model_tier" not in df.columns:
            df["model_tier"] = ""
        frames.append(df)
        print(f"    Loaded {p.name}: {len(df):,} rows, "
              f"models={df['model'].nunique()}, "
              f"prompts={df['system_prompt_name'].nunique()}")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# COMPUTE SHIFTS (core metric)
# ─────────────────────────────────────────────────────────────
def compute_shift(df, group_cols=None):
    """
    Compute ad-induced brand selection shift.
    Returns one row per group, with baseline rate, ad rate, shift.
    group_cols: extra columns to group by (e.g., ["model", "system_prompt_name"])
    """
    bl = df[df["condition"] == "baseline"]
    ad = df[df["condition"] != "baseline"]
    if ad.empty:
        return pd.DataFrame()

    base_groups = ["scenario_id", "ad_brand"]
    if group_cols:
        base_groups = group_cols + base_groups

    records = []
    for brand in sorted(ad["ad_brand"].dropna().unique()):
        ab = ad[ad["ad_brand"] == brand]

        # Determine subgroups
        if group_cols:
            sub_groups = ab.groupby(group_cols)
        else:
            sub_groups = [("all", ab)]

        for grp_key, sub_ad in sub_groups:
            if isinstance(grp_key, str):
                grp_dict = {"group": grp_key} if not group_cols else {group_cols[0]: grp_key}
            elif group_cols:
                grp_dict = dict(zip(group_cols, grp_key)) if isinstance(grp_key, tuple) else {group_cols[0]: grp_key}
            else:
                grp_dict = {}

            ref = sub_ad.iloc[0]
            adv_opt = ref["advertised_option"]
            sid = ref["scenario_id"]
            is_rx = ref.get("is_rx", True)

            # Match baseline on same grouping
            bl_filter = bl["scenario_id"] == sid
            if group_cols:
                for gc in group_cols:
                    if gc in bl.columns and gc in ref.index:
                        bl_filter = bl_filter & (bl[gc] == ref[gc])
            bl_sub = bl[bl_filter]

            bl_rate = (bl_sub["choice"] == adv_opt).mean() if (not bl_sub.empty and adv_opt) else 0
            ad_rate = sub_ad["chose_advertised"].mean() if sub_ad["chose_advertised"].notna().any() else 0

            rec = {
                **grp_dict,
                "brand": brand,
                "scenario": sid,
                "type": "Rx" if is_rx else "Wellness",
                "answer_category": ref.get("answer_category", ""),
                "evidence_tier": ref.get("evidence_tier", ""),
                "therapeutic_area": str(ref.get("therapeutic_area", ""))[:30],
                "n_baseline": len(bl_sub),
                "n_ad": len(sub_ad),
                "bl_rate": bl_rate,
                "ad_rate": ad_rate,
                "shift": ad_rate - bl_rate,
            }
            records.append(rec)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ─────────────────────────────────────────────────────────────
def _header(fig, title, sub=""):
    fig.text(0.04, 0.97, title, size=17, weight="bold", va="top", color=TXT)
    if sub:
        fig.text(0.04, 0.935, sub, size=9.5, va="top", color=SUB)

def _wm(fig, info=""):
    fig.text(0.98, 0.008, f"Ad-verse Effects v5  |  {info}",
             size=7, ha="right", va="bottom", color=SUB, alpha=0.45)

def _pct(ax, axis="y"):
    fmt = mtick.PercentFormatter(1.0, decimals=0)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)


# ═══════════════════════════════════════════════════════════════
# FIGURE 1 -- OVERALL BRAND SHIFT (aggregated across all)
# ═══════════════════════════════════════════════════════════════
def fig1_overall_shift(df, info=""):
    """THE headline figure: per-brand shift aggregated across all models/prompts."""
    shifts = compute_shift(df)
    if shifts.empty:
        return None

    # Aggregate across variants/repeats per brand
    agg = shifts.groupby(["brand", "scenario", "type", "answer_category"]).agg(
        bl_rate=("bl_rate", "mean"),
        ad_rate=("ad_rate", "mean"),
        shift=("shift", "mean"),
    ).reset_index().sort_values("shift", ascending=True)

    h = max(7, len(agg) * 0.35 + 3)
    fig, axes = plt.subplots(1, 2, figsize=(18, h),
                              gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.22, right=0.97, wspace=0.35)
    _header(fig, "Figure 1 — Overall Ad-Induced Brand Selection Shift",
            "Aggregated across all models and system prompts")
    _wm(fig, info)

    y = np.arange(len(agg))
    labels = [f"{row['scenario']}  {row['brand']}" for _, row in agg.iterrows()]

    # Panel A: Paired dot plot
    ax = axes[0]
    for i, (_, row) in enumerate(agg.iterrows()):
        color = RED if row["shift"] > 0.02 else (AMBER if row["shift"] > 0.005 else SLATE)
        ax.plot([row["bl_rate"], row["ad_rate"]], [i, i],
                color=color, linewidth=1.5, alpha=0.6, zorder=3)

    ax.scatter(agg["bl_rate"], y, color=BLUE, s=50, zorder=5,
               edgecolors=WHITE, linewidth=0.8, label="Baseline")
    ax.scatter(agg["ad_rate"], y, color=RED, s=50, zorder=5,
               edgecolors=WHITE, linewidth=0.8, label="With Ad")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, size=8)
    ax.set_xlabel("Brand Selection Rate")
    ax.set_title("A  Baseline vs Ad Selection Rate", weight="bold", loc="left")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.05, 1.15)
    _pct(ax, "x")

    # Panel B: Shift bars
    ax2 = axes[1]
    colors = [RED if s > 0.02 else (AMBER if s > 0.005 else SLATE) for s in agg["shift"]]
    ax2.barh(y, agg["shift"], color=colors, edgecolor=WHITE, height=0.65)
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(y))
    ax2.axvline(0, color=TXT, lw=0.7, alpha=0.4)
    ax2.set_xlabel("Shift (Ad − Baseline)")
    ax2.set_title("B  Selection Shift", weight="bold", loc="left")

    for i, (_, row) in enumerate(agg.iterrows()):
        if abs(row["shift"]) > 0.005:
            side = 0.008 * (1 if row["shift"] >= 0 else -1)
            ax2.text(row["shift"] + side, i, f"{row['shift']:+.0%}",
                     va="center", size=8, weight="bold",
                     color=RED if row["shift"] > 0.02 else SUB)
    _pct(ax2, "x")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 2 -- SHIFT BY SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════
def fig2_by_prompt(df, info=""):
    """Does the system prompt persona affect ad susceptibility?"""
    prompts = sorted(df["system_prompt_name"].dropna().unique())
    if len(prompts) < 2:
        return None

    bl = df[df["condition"] == "baseline"]
    ad = df[df["condition"] != "baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(top=0.85, bottom=0.14, left=0.08, right=0.97, wspace=0.30)
    _header(fig, "Figure 2 — Ad Susceptibility by System Prompt Persona",
            "Does framing the model as 'physician' vs 'customer service' change ad influence?")
    _wm(fig, info)

    # A: Mean shift per prompt
    ax = axes[0]
    x = np.arange(len(prompts))
    mean_shifts = []
    se_shifts = []
    for sp in prompts:
        sp_shifts = compute_shift(ad[ad["system_prompt_name"] == sp].append(
            bl[bl["system_prompt_name"] == sp]) if False else df[df["system_prompt_name"] == sp])
        if sp_shifts.empty:
            mean_shifts.append(0)
            se_shifts.append(0)
        else:
            mean_shifts.append(sp_shifts["shift"].mean())
            se_shifts.append(sp_shifts["shift"].std() / np.sqrt(len(sp_shifts)) if len(sp_shifts) > 1 else 0)

    colors = [PROMPT_COLORS.get(sp, SLATE) for sp in prompts]
    bars = ax.bar(x, mean_shifts, color=colors, edgecolor=WHITE, width=0.6,
                  yerr=se_shifts, capsize=4, error_kw={"lw": 1.2, "color": TXT})
    for b, v in zip(bars, mean_shifts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005 if v >= 0 else v - 0.015,
                f"{v:+.1%}", ha="center", size=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS.get(sp, sp) for sp in prompts], size=10)
    ax.set_ylabel("Mean Brand Selection Shift")
    ax.axhline(0, color=TXT, lw=0.7, alpha=0.4)
    ax.set_title("A  Mean Shift per Persona", weight="bold", loc="left")
    _pct(ax)

    # B: Accuracy per prompt (baseline)
    ax = axes[1]
    bl_acc = []
    ad_acc = []
    for sp in prompts:
        sb = bl[bl["system_prompt_name"] == sp]
        sa = ad[ad["system_prompt_name"] == sp]
        bl_acc.append(sb["is_correct"].mean() if not sb.empty else 0)
        ad_acc.append(sa["is_correct"].mean() if not sa.empty else 0)

    w = 0.3
    ax.bar(x - w / 2, bl_acc, w, color=BLUE, label="Baseline", edgecolor=WHITE)
    ax.bar(x + w / 2, ad_acc, w, color=RED, label="With Ad", edgecolor=WHITE)
    for i in range(len(prompts)):
        ax.text(i - w / 2, bl_acc[i] + 0.015, f"{bl_acc[i]:.0%}", ha="center", size=9, weight="bold")
        ax.text(i + w / 2, ad_acc[i] + 0.015, f"{ad_acc[i]:.0%}", ha="center", size=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS.get(sp, sp) for sp in prompts], size=10)
    ax.set_ylabel("Correct Answer Rate")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title("B  Accuracy by Persona", weight="bold", loc="left")
    ax.set_ylim(0, 1.15)
    _pct(ax)
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 3 -- SHIFT BY MODEL
# ═══════════════════════════════════════════════════════════════
def fig3_by_model(df, info=""):
    """Do nano/mini models differ from flagship?"""
    models = sorted(df["model"].dropna().unique())
    if len(models) < 2:
        return None

    bl = df[df["condition"] == "baseline"]
    ad = df[df["condition"] != "baseline"]

    h = max(6, len(models) * 0.5 + 2)
    fig, axes = plt.subplots(1, 2, figsize=(18, h))
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.18, right=0.97, wspace=0.30)
    _header(fig, "Figure 3 — Ad Susceptibility by Model",
            "Shift and accuracy across model sizes (nano → flagship)")
    _wm(fig, info)

    y = np.arange(len(models))

    # A: Mean shift per model
    ax = axes[0]
    mean_shifts = []
    for m in models:
        ms = compute_shift(df[df["model"] == m])
        mean_shifts.append(ms["shift"].mean() if not ms.empty else 0)

    colors = [RED if s > 0.05 else (AMBER if s > 0.02 else BLUE) for s in mean_shifts]
    ax.barh(y, mean_shifts, color=colors, edgecolor=WHITE, height=0.6)
    ax.axvline(0, color=TXT, lw=0.7, alpha=0.4)
    for i, v in enumerate(mean_shifts):
        side = 0.005 * (1 if v >= 0 else -1)
        ax.text(v + side, i, f"{v:+.1%}", va="center", size=9, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(models, size=9)
    ax.set_xlabel("Mean Brand Selection Shift")
    ax.set_title("A  Shift by Model", weight="bold", loc="left")
    _pct(ax, "x")

    # B: Accuracy per model
    ax = axes[1]
    bl_acc = []
    ad_acc = []
    for m in models:
        mb = bl[bl["model"] == m]
        ma = ad[ad["model"] == m]
        bl_acc.append(mb["is_correct"].mean() if not mb.empty else 0)
        ad_acc.append(ma["is_correct"].mean() if not ma.empty else 0)

    w = 0.3
    ax.barh(y - w / 2, bl_acc, w, color=BLUE, label="Baseline", edgecolor=WHITE)
    ax.barh(y + w / 2, ad_acc, w, color=RED, label="With Ad", edgecolor=WHITE)
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(models))
    ax.set_xlabel("Correct Answer Rate")
    ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax.set_title("B  Accuracy by Model", weight="bold", loc="left")
    _pct(ax, "x")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 4 -- SHIFT BY ANSWER CATEGORY × PROMPT
# ═══════════════════════════════════════════════════════════════
def fig4_category_x_prompt(df, info=""):
    """Where do ads bite hardest? Category × prompt interaction."""
    cats = ["clear_preference", "equipoise", "neither_correct"]
    cat_labels = ["Clear Preference", "Equipoise", "Neither Correct"]
    prompts = sorted(df["system_prompt_name"].dropna().unique())

    if len(prompts) < 2:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.06, right=0.97, wspace=0.30)
    _header(fig, "Figure 4 — Shift by Clinical Scenario Type × System Prompt",
            "Equipoise scenarios are most susceptible; does persona amplify this?")
    _wm(fig, info)

    for i, (cat, clabel) in enumerate(zip(cats, cat_labels)):
        ax = axes[i]
        cat_df = df[df["answer_category"] == cat]
        if cat_df.empty:
            continue

        x = np.arange(len(prompts))
        shifts = []
        for sp in prompts:
            sp_df = cat_df[cat_df["system_prompt_name"] == sp]
            ms = compute_shift(sp_df)
            shifts.append(ms["shift"].mean() if not ms.empty else 0)

        colors = [PROMPT_COLORS.get(sp, SLATE) for sp in prompts]
        bars = ax.bar(x, shifts, color=colors, edgecolor=WHITE, width=0.55)
        for b, v in zip(bars, shifts):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005 if v >= 0 else v - 0.015,
                    f"{v:+.0%}", ha="center", size=10, weight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([PROMPT_LABELS.get(sp, sp)[:12] for sp in prompts],
                          size=8, rotation=20, ha="right")
        ax.axhline(0, color=TXT, lw=0.7, alpha=0.4)
        ax.set_ylabel("Mean Shift" if i == 0 else "")
        ax.set_title(f"{clabel}", weight="bold", loc="left", size=12)
        _pct(ax)
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 5 -- WELLNESS ENDORSEMENT
# ═══════════════════════════════════════════════════════════════
def fig5_wellness(df, info=""):
    """Wellness supplement endorsement: by prompt and overall."""
    well = df[df["is_rx"] == False]
    if well.empty:
        return None

    bl = well[well["condition"] == "baseline"]
    ad = well[well["condition"] != "baseline"]
    prompts = sorted(well["system_prompt_name"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(top=0.85, bottom=0.14, left=0.10, right=0.97, wspace=0.30)
    _header(fig, "Figure 5 — Wellness Supplement Endorsement",
            "% of responses recommending the supplement (A) vs evidence-based advice (B)")
    _wm(fig, info)

    # A: By system prompt
    ax = axes[0]
    x = np.arange(len(prompts))
    bl_rates = []
    ad_rates = []
    for sp in prompts:
        sb = bl[bl["system_prompt_name"] == sp]
        sa = ad[ad["system_prompt_name"] == sp]
        bl_rates.append((sb["choice"] == "A").mean() if not sb.empty else 0)
        ad_rates.append((sa["choice"] == "A").mean() if not sa.empty else 0)

    w = 0.3
    ax.bar(x - w / 2, bl_rates, w, color=BLUE, label="Baseline", edgecolor=WHITE)
    ax.bar(x + w / 2, ad_rates, w, color=RED, label="With Ad", edgecolor=WHITE)
    for i in range(len(prompts)):
        ax.text(i - w / 2, bl_rates[i] + 0.01, f"{bl_rates[i]:.0%}", ha="center", size=9, weight="bold")
        ax.text(i + w / 2, ad_rates[i] + 0.01, f"{ad_rates[i]:.0%}", ha="center", size=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS.get(sp, sp) for sp in prompts], size=10)
    ax.set_ylabel("Supplement Endorsement Rate")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title("A  By Persona", weight="bold", loc="left")
    _pct(ax)

    # B: Overall
    ax = axes[1]
    bl_overall = (bl["choice"] == "A").mean() if not bl.empty else 0
    ad_overall = (ad["choice"] == "A").mean() if not ad.empty else 0
    bars = ax.bar(["Baseline", "With Ad"], [bl_overall, ad_overall],
                  color=[BLUE, RED], width=0.5, edgecolor=WHITE, linewidth=1.5)
    for b, v in zip(bars, [bl_overall, ad_overall]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                f"{v:.1%}", ha="center", weight="bold", size=14)
    d = ad_overall - bl_overall
    ax.set_title("B  Overall", weight="bold", loc="left")
    ax.set_ylabel("Endorsement Rate")
    ax.set_ylim(0, max(0.3, max(bl_overall, ad_overall) * 2))
    if abs(d) > 0.001:
        ax.text(0.5, max(bl_overall, ad_overall) * 1.5 + 0.02,
                f"Shift: {d:+.1%}", ha="center", size=13, weight="bold",
                color=RED if d > 0 else GREEN, transform=ax.get_xaxis_transform())
    _pct(ax)
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 6 -- MODEL × PROMPT HEATMAP
# ═══════════════════════════════════════════════════════════════
def fig6_model_prompt_heatmap(df, info=""):
    """Interaction: model × prompt mean shift."""
    models = sorted(df["model"].dropna().unique())
    prompts = sorted(df["system_prompt_name"].dropna().unique())

    if len(models) < 2 or len(prompts) < 2:
        return None

    mat = np.full((len(models), len(prompts)), np.nan)
    for i, m in enumerate(models):
        for j, sp in enumerate(prompts):
            sub = df[(df["model"] == m) & (df["system_prompt_name"] == sp)]
            ms = compute_shift(sub)
            if not ms.empty:
                mat[i, j] = ms["shift"].mean()

    h = max(5, len(models) * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(max(8, len(prompts) * 2.5 + 2), h))
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.18, right=0.90)
    _header(fig, "Figure 6 — Model × Prompt Interaction",
            "Mean brand selection shift: which model/persona combinations are most susceptible?")
    _wm(fig, info)

    vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.5
    cmap = LinearSegmentedColormap.from_list("shift", [BLUE, WHITE, RED])
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels([PROMPT_LABELS.get(sp, sp) for sp in prompts], rotation=20, ha="right", size=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, size=9)

    for i in range(len(models)):
        for j in range(len(prompts)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0%}", ha="center", va="center",
                        size=10, weight="bold", color=TXT if abs(v) < vmax * 0.5 else WHITE)

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Mean Shift", size=9)
    cb.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 7 -- ACCURACY HEATMAP (secondary)
# ═══════════════════════════════════════════════════════════════
def fig7_accuracy_heatmap(df, info=""):
    """Scenario × condition accuracy heatmap."""
    sids = sorted(df["scenario_id"].unique())
    conds = sorted(df["condition"].unique())
    if "baseline" in conds:
        conds = ["baseline"] + [c for c in conds if c != "baseline"]

    mat = np.full((len(sids), len(conds)), np.nan)
    for i, s in enumerate(sids):
        for j, c in enumerate(conds):
            sub = df[(df["scenario_id"] == s) & (df["condition"] == c)]
            if not sub.empty:
                mat[i, j] = sub["is_correct"].mean()

    h = max(6, len(sids) * 0.38 + 2)
    fig, ax = plt.subplots(figsize=(max(8, len(conds) * 1.5 + 3), h))
    fig.subplots_adjust(top=0.90, bottom=0.13, left=0.09, right=0.92)
    _header(fig, "Figure 7 — Correct Answer Rate Heatmap (Secondary)",
            "Scenario × Condition")
    _wm(fig, info)

    cmap = LinearSegmentedColormap.from_list(
        "acc", [(0, RED), (0.5, "#FEF3C7"), (0.85, "#D1FAE5"), (1, GREEN)])
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=30, ha="right", size=9)
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels(sids, size=9)
    for i in range(len(sids)):
        for j in range(len(conds)):
            v = mat[i, j]
            if not np.isnan(v):
                tc = WHITE if v < 0.4 else TXT
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        size=8, weight="bold", color=tc)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Correct %", size=9)
    cb.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    return fig


# ═══════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════
def report(df):
    bl = df[df["condition"] == "baseline"]
    ad = df[df["condition"] != "baseline"]
    models = sorted(df["model"].dropna().unique())
    prompts = sorted(df["system_prompt_name"].dropna().unique())

    W = 80
    print("\n" + "=" * W)
    print("  AD-VERSE EFFECTS v5 — RESULTS")
    print(f"  Models: {', '.join(models)}")
    print(f"  Prompts: {', '.join(prompts)}")
    print("=" * W)

    print(f"\n  Total calls: {len(df):,}  |  "
          f"{df['scenario_id'].nunique()} scenarios  |  "
          f"{len(models)} model(s)  |  {len(prompts)} prompt(s)")

    # ──────────────────────────────────────────────
    # TABLE 1: OVERALL BRAND SELECTION SHIFT (AGGREGATED)
    # ──────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  TABLE 1: OVERALL BRAND SELECTION SHIFT — AGGREGATED (PRIMARY)")
    print(f"  {'='*W}")

    shifts_all = compute_shift(df)
    if not shifts_all.empty:
        agg = shifts_all.groupby(["brand", "scenario", "type", "answer_category"]).agg(
            bl=("bl_rate", "mean"), ad=("ad_rate", "mean"), shift=("shift", "mean"),
        ).reset_index().sort_values("shift", ascending=False)

        print(f"  {'Brand':<22} {'Scen':<6} {'Type':<9} {'Category':<20} "
              f"{'Baseline':>9} {'With Ad':>9} {'Shift':>9}")
        print(f"  {'-'*22} {'-'*6} {'-'*9} {'-'*20} {'-'*9} {'-'*9} {'-'*9}")
        for _, r in agg.iterrows():
            flag = " ***" if abs(r["shift"]) > 0.1 else (" **" if abs(r["shift"]) > 0.05 else "")
            print(f"  {r['brand']:<22} {r['scenario']:<6} {r['type']:<9} "
                  f"{r['answer_category']:<20} "
                  f"{r['bl']:>8.0%} {r['ad']:>8.0%} {r['shift']:>+8.0%}{flag}")

        mean_shift = agg["shift"].mean()
        print(f"\n  Overall mean shift: {mean_shift:+.1%}")

    # ──────────────────────────────────────────────
    # TABLE 2: SHIFT BY SYSTEM PROMPT
    # ──────────────────────────────────────────────
    if len(prompts) > 1:
        print(f"\n  {'='*W}")
        print("  TABLE 2: BRAND SHIFT BY SYSTEM PROMPT CONDITION")
        print(f"  {'='*W}")
        print(f"  {'Prompt':<22} {'n brands':>10} {'Mean Shift':>12} {'Max Shift':>12} {'Accuracy BL':>12} {'Accuracy Ad':>12}")
        print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for sp in prompts:
            sp_df = df[df["system_prompt_name"] == sp]
            sp_shifts = compute_shift(sp_df)
            sp_bl = bl[bl["system_prompt_name"] == sp]
            sp_ad = ad[ad["system_prompt_name"] == sp]
            if sp_shifts.empty:
                continue
            ms = sp_shifts["shift"].mean()
            mx = sp_shifts["shift"].max()
            bl_acc = sp_bl["is_correct"].mean() if not sp_bl.empty else 0
            ad_acc = sp_ad["is_correct"].mean() if not sp_ad.empty else 0
            label = PROMPT_LABELS.get(sp, sp)
            print(f"  {label:<22} {len(sp_shifts):>10} {ms:>+11.1%} {mx:>+11.1%} "
                  f"{bl_acc:>11.1%} {ad_acc:>11.1%}")

    # ──────────────────────────────────────────────
    # TABLE 3: SHIFT BY MODEL
    # ──────────────────────────────────────────────
    if len(models) > 1:
        print(f"\n  {'='*W}")
        print("  TABLE 3: BRAND SHIFT BY MODEL")
        print(f"  {'='*W}")
        print(f"  {'Model':<35} {'Tier':<10} {'Mean Shift':>12} {'Max Shift':>12} {'BL Acc':>10} {'Ad Acc':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

        for m in models:
            m_df = df[df["model"] == m]
            m_shifts = compute_shift(m_df)
            m_bl = bl[bl["model"] == m]
            m_ad = ad[ad["model"] == m]
            tier = m_df["model_tier"].iloc[0] if "model_tier" in m_df and not m_df.empty else ""
            if m_shifts.empty:
                continue
            ms = m_shifts["shift"].mean()
            mx = m_shifts["shift"].max()
            bl_acc = m_bl["is_correct"].mean() if not m_bl.empty else 0
            ad_acc = m_ad["is_correct"].mean() if not m_ad.empty else 0
            print(f"  {m:<35} {str(tier):<10} {ms:>+11.1%} {mx:>+11.1%} "
                  f"{bl_acc:>9.1%} {ad_acc:>9.1%}")

    # ──────────────────────────────────────────────
    # TABLE 4: SHIFT BY ANSWER CATEGORY
    # ──────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  TABLE 4: SHIFT BY CLINICAL SCENARIO TYPE")
    print(f"  {'='*W}")
    print(f"  {'Category':<25} {'n':>5} {'Mean Shift':>12} {'Max Shift':>12}")
    print(f"  {'-'*25} {'-'*5} {'-'*12} {'-'*12}")

    for cat in ["clear_preference", "equipoise", "neither_correct"]:
        cat_shifts = shifts_all[shifts_all["answer_category"] == cat] if not shifts_all.empty else pd.DataFrame()
        if cat_shifts.empty:
            continue
        ms = cat_shifts["shift"].mean()
        mx = cat_shifts["shift"].max()
        print(f"  {cat:<25} {len(cat_shifts):>5} {ms:>+11.1%} {mx:>+11.1%}")

    # ──────────────────────────────────────────────
    # TABLE 5: WELLNESS SUPPLEMENT ENDORSEMENT
    # ──────────────────────────────────────────────
    well_bl = bl[bl["is_rx"] == False]
    well_ad = ad[ad["is_rx"] == False]
    if not well_bl.empty:
        print(f"\n  {'='*W}")
        print("  TABLE 5: WELLNESS SUPPLEMENT ENDORSEMENT")
        print(f"  {'='*W}")

        bl_total = (well_bl["choice"] == "A").mean()
        ad_total = (well_ad["choice"] == "A").mean() if not well_ad.empty else 0
        print(f"  Overall:  Baseline {bl_total:.1%}  →  With Ad {ad_total:.1%}  "
              f"(shift {ad_total - bl_total:+.1%})")

        if len(prompts) > 1:
            print(f"\n  {'Prompt':<22} {'BL Endorse':>12} {'Ad Endorse':>12} {'Shift':>10}")
            print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")
            for sp in prompts:
                sb = well_bl[well_bl["system_prompt_name"] == sp]
                sa = well_ad[well_ad["system_prompt_name"] == sp]
                br = (sb["choice"] == "A").mean() if not sb.empty else 0
                ar = (sa["choice"] == "A").mean() if not sa.empty else 0
                label = PROMPT_LABELS.get(sp, sp)
                print(f"  {label:<22} {br:>11.1%} {ar:>11.1%} {ar-br:>+9.1%}")

    # ──────────────────────────────────────────────
    # TABLE 6: ACCURACY (SECONDARY)
    # ──────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  TABLE 6: ACCURACY — SECONDARY ANALYSIS")
    print(f"  {'='*W}")

    bg = bl["is_correct"].mean() if not bl.empty else 0
    ag = ad["is_correct"].mean() if not ad.empty else 0
    bg_p, bg_lo, bg_hi = wilson_ci(int(bl["is_correct"].sum()), len(bl)) if not bl.empty else (0, 0, 0)
    ag_p, ag_lo, ag_hi = wilson_ci(int(ad["is_correct"].sum()), len(ad)) if not ad.empty else (0, 0, 0)

    print(f"  Baseline:  {bg:.1%}  95% CI [{bg_lo:.1%}, {bg_hi:.1%}]  (n={len(bl):,})")
    print(f"  With Ad:   {ag:.1%}  95% CI [{ag_lo:.1%}, {ag_hi:.1%}]  (n={len(ad):,})")
    print(f"  Delta:     {ag - bg:+.1%}")

    print()


# ═══════════════════════════════════════════════════════════════
# SUMMARY EXCEL
# ═══════════════════════════════════════════════════════════════
def write_summary(df, out_path):
    bl = df[df["condition"] == "baseline"]
    ad = df[df["condition"] != "baseline"]

    # Sheet 1: Overall Brand Shift
    shifts_all = compute_shift(df)
    if not shifts_all.empty:
        overall = shifts_all.groupby(["brand", "scenario", "type", "answer_category"]).agg(
            bl_pct=("bl_rate", lambda x: round(x.mean() * 100, 1)),
            ad_pct=("ad_rate", lambda x: round(x.mean() * 100, 1)),
            shift_pp=("shift", lambda x: round(x.mean() * 100, 1)),
        ).reset_index().sort_values("shift_pp", ascending=False)
        overall.columns = ["Brand", "Scenario", "Type", "Answer Category",
                           "Baseline %", "With Ad %", "Shift (pp)"]
    else:
        overall = pd.DataFrame()

    # Sheet 2: Shift by System Prompt
    prompts = sorted(df["system_prompt_name"].dropna().unique())
    prompt_rows = []
    for sp in prompts:
        sp_df = df[df["system_prompt_name"] == sp]
        sp_shifts = compute_shift(sp_df)
        sp_bl = bl[bl["system_prompt_name"] == sp]
        sp_ad = ad[ad["system_prompt_name"] == sp]
        if sp_shifts.empty:
            continue
        prompt_rows.append({
            "System Prompt": PROMPT_LABELS.get(sp, sp),
            "n brand-scenarios": len(sp_shifts),
            "Mean Shift (pp)": round(sp_shifts["shift"].mean() * 100, 1),
            "Max Shift (pp)": round(sp_shifts["shift"].max() * 100, 1),
            "Baseline Accuracy %": round(sp_bl["is_correct"].mean() * 100, 1) if not sp_bl.empty else 0,
            "Ad Accuracy %": round(sp_ad["is_correct"].mean() * 100, 1) if not sp_ad.empty else 0,
        })
    prompt_df = pd.DataFrame(prompt_rows)

    # Sheet 3: Shift by Model
    models = sorted(df["model"].dropna().unique())
    model_rows = []
    for m in models:
        m_df = df[df["model"] == m]
        m_shifts = compute_shift(m_df)
        m_bl = bl[bl["model"] == m]
        m_ad = ad[ad["model"] == m]
        tier = m_df["model_tier"].iloc[0] if "model_tier" in m_df and not m_df.empty else ""
        if m_shifts.empty:
            continue
        model_rows.append({
            "Model": m,
            "Tier": tier,
            "n brand-scenarios": len(m_shifts),
            "Mean Shift (pp)": round(m_shifts["shift"].mean() * 100, 1),
            "Max Shift (pp)": round(m_shifts["shift"].max() * 100, 1),
            "Baseline Accuracy %": round(m_bl["is_correct"].mean() * 100, 1) if not m_bl.empty else 0,
            "Ad Accuracy %": round(m_ad["is_correct"].mean() * 100, 1) if not m_ad.empty else 0,
        })
    model_df = pd.DataFrame(model_rows)

    # Sheet 4: Shift by Answer Category
    cat_rows = []
    for cat in ["clear_preference", "equipoise", "neither_correct"]:
        cat_shifts = shifts_all[shifts_all["answer_category"] == cat] if not shifts_all.empty else pd.DataFrame()
        if cat_shifts.empty:
            continue
        cat_rows.append({
            "Category": cat,
            "n brand-scenarios": len(cat_shifts),
            "Mean Shift (pp)": round(cat_shifts["shift"].mean() * 100, 1),
            "Max Shift (pp)": round(cat_shifts["shift"].max() * 100, 1),
        })
    cat_df = pd.DataFrame(cat_rows)

    # Sheet 5: Model × Prompt interaction
    interaction_rows = []
    for m in models:
        for sp in prompts:
            sub = df[(df["model"] == m) & (df["system_prompt_name"] == sp)]
            ms = compute_shift(sub)
            sub_bl = bl[(bl["model"] == m) & (bl["system_prompt_name"] == sp)]
            sub_ad = ad[(ad["model"] == m) & (ad["system_prompt_name"] == sp)]
            if ms.empty:
                continue
            interaction_rows.append({
                "Model": m,
                "System Prompt": PROMPT_LABELS.get(sp, sp),
                "Mean Shift (pp)": round(ms["shift"].mean() * 100, 1),
                "Max Shift (pp)": round(ms["shift"].max() * 100, 1),
                "BL Accuracy %": round(sub_bl["is_correct"].mean() * 100, 1) if not sub_bl.empty else 0,
                "Ad Accuracy %": round(sub_ad["is_correct"].mean() * 100, 1) if not sub_ad.empty else 0,
            })
    interaction_df = pd.DataFrame(interaction_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        if not overall.empty:
            overall.to_excel(w, index=False, sheet_name="Overall_Shift")
        if not prompt_df.empty:
            prompt_df.to_excel(w, index=False, sheet_name="By_Prompt")
        if not model_df.empty:
            model_df.to_excel(w, index=False, sheet_name="By_Model")
        if not cat_df.empty:
            cat_df.to_excel(w, index=False, sheet_name="By_Category")
        if not interaction_df.empty:
            interaction_df.to_excel(w, index=False, sheet_name="Model_x_Prompt")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    _style()

    print("\n" + "=" * 70)
    print("  AD-VERSE EFFECTS v5 — ANALYSIS & VISUALIZATION")
    print("  Supports multi-model, multi-prompt aggregation")
    print("=" * 70)

    # Collect input files
    paths = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            p = Path(arg).expanduser()
            if p.is_dir():
                paths.extend(sorted(p.glob("adverse_*.xlsx")))
            elif p.exists():
                paths.append(p)
    else:
        raw = input("\n  Path to output file(s) or directory: ").strip().strip("'\"")
        if not raw:
            # Auto-detect
            cwd = Path(".")
            hits = sorted(Path("results/main").glob("adverse_*.xlsx"), key=lambda x: x.stat().st_mtime)
            if hits:
                paths = hits
                print(f"  Auto-detected {len(paths)} file(s)")
            else:
                print("  No files found. Exiting.")
                return
        else:
            p = Path(raw).expanduser()
            if p.is_dir():
                paths = sorted(p.glob("adverse_*.xlsx"))
            elif p.exists():
                paths = [p]
            elif "*" in raw:
                paths = sorted(Path(".").glob(raw))

    if not paths:
        print("  No valid files found. Exiting.")
        return

    print(f"\n  Loading {len(paths)} file(s)...")
    df = load_all(paths)
    print(f"\n  Combined: {len(df):,} rows  |  "
          f"{df['scenario_id'].nunique()} scenarios  |  "
          f"{df['model'].nunique()} model(s)  |  "
          f"{df['system_prompt_name'].nunique()} prompt(s)")

    # Info string for watermarks
    models = sorted(df["model"].dropna().unique())
    info = f"{len(models)} model(s), {df['system_prompt_name'].nunique()} prompts"

    # Console report
    report(df)

    # Figures
    print("  Generating figures...")
    figure_funcs = [
        ("fig1_overall_shift",       fig1_overall_shift,       (df, info)),
        ("fig2_by_prompt",           fig2_by_prompt,           (df, info)),
        ("fig3_by_model",            fig3_by_model,            (df, info)),
        ("fig4_category_x_prompt",   fig4_category_x_prompt,   (df, info)),
        ("fig5_wellness",            fig5_wellness,            (df, info)),
        ("fig6_model_prompt_heatmap", fig6_model_prompt_heatmap, (df, info)),
        ("fig7_accuracy_heatmap",    fig7_accuracy_heatmap,    (df, info)),
    ]

    figs = []
    for name, fn, args in figure_funcs:
        try:
            f = fn(*args)
            if f is not None:
                figs.append((name, f))
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")

    # Output directory
    out_dir = paths[0].parent if paths else Path(".")
    stem = "adverse_combined" if len(paths) > 1 else paths[0].stem

    # Multi-page PDF
    pdf_path = out_dir / f"{stem}_figures.pdf"
    with PdfPages(pdf_path) as pdf:
        for _, f in figs:
            pdf.savefig(f, dpi=200)
            plt.close(f)
    print(f"    Saved {pdf_path.name}  ({len(figs)} figures)")

    # Individual PNGs
    png_dir = out_dir / f"{stem}_plots"
    png_dir.mkdir(exist_ok=True)
    for name, fn, args in figure_funcs:
        try:
            f = fn(*args)
            if f is not None:
                f.savefig(png_dir / f"{name}.png", dpi=200,
                          bbox_inches="tight", facecolor=f.get_facecolor())
                plt.close(f)
        except Exception:
            pass
    print(f"    Saved PNGs to {png_dir.name}/")

    # Summary Excel
    xlsx_path = out_dir / f"{stem}_summary.xlsx"
    write_summary(df, xlsx_path)
    print(f"    Saved {xlsx_path.name}")

    print(f"\n  All output in: {out_dir}\n")


if __name__ == "__main__":
    main()
