#!/usr/bin/env python3
"""
Ad-verse Effects: Accuracy Cost Analysis (Experiment 3)
========================================================
Does ad-induced brand preference degrade clinical accuracy?

Primary metric:
  Accuracy cost = P(correct answer | baseline) - P(correct answer | ad)

Secondary metrics:
  - Shift toward advertised brand (A or B) at expense of correct (C)
  - Interaction: system prompt x scenario x model
  - Per-scenario accuracy erosion

Accepts one or more augmentation pipeline output files.

Usage:
    python analysis_augmentation.py [file1.xlsx] [file2.xlsx] ...
    python analysis_augmentation.py                    # auto-detect
"""

import sys, math, json
from pathlib import Path
from typing import List, Tuple

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

PROMPT_COLORS = {
    "physician": BLUE, "helpful_ai": AMBER,
    "customer_service": RED, "no_persona": SLATE,
}
PROMPT_LABELS = {
    "physician": "Physician", "helpful_ai": "Helpful AI",
    "customer_service": "Customer Svc", "no_persona": "No Persona",
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
# LOAD
# ─────────────────────────────────────────────────────────────
def load_one(path: Path) -> pd.DataFrame:
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
    frames = []
    for p in paths:
        df = load_one(p)
        for col in ["is_correct", "chose_advertised", "chose_correct_nonadvertised",
                     "chose_advertised_specific", "parse_ok", "chose_A", "chose_B", "chose_C"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("boolean")
        if "system_prompt_name" not in df.columns:
            df["system_prompt_name"] = "physician"
        if "model" not in df.columns:
            name = p.stem
            model_str = name.replace("aug_", "").rsplit("_", 2)[0]
            df["model"] = model_str
        if "model_tier" not in df.columns:
            df["model_tier"] = ""
        frames.append(df)
        print(f"    Loaded {p.name}: {len(df):,} rows, "
              f"models={df['model'].nunique()}, prompts={df['system_prompt_name'].nunique()}")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# COMPUTE ACCURACY COST (core metric)
# ─────────────────────────────────────────────────────────────
def compute_accuracy_cost(df, group_cols=None):
    """
    Per scenario (or per group), compute:
      bl_accuracy: P(correct | baseline)
      ad_accuracy: P(correct | ad)
      accuracy_cost: bl_accuracy − ad_accuracy  (positive = ads hurt accuracy)
      bl_chose_C: P(C | baseline)
      ad_chose_C: P(C | ad)
      correct_erosion: bl_chose_C − ad_chose_C
    """
    bl = df[df["condition"] == "baseline"]
    ad = df[df["condition"] != "baseline"]
    if ad.empty or bl.empty:
        return pd.DataFrame()

    base_groups = ["scenario_id"]
    if group_cols:
        base_groups = group_cols + base_groups

    records = []
    group_keys = ad.groupby(base_groups).groups.keys() if len(base_groups) > 1 else ad["scenario_id"].unique()

    for key in (ad.groupby(base_groups) if len(base_groups) > 1 else [(s, ad[ad["scenario_id"] == s]) for s in ad["scenario_id"].unique()]):
        if isinstance(key, tuple):
            grp_key, sub_ad = key
        else:
            grp_key, sub_ad = key

        if isinstance(grp_key, str):
            grp_dict = {"scenario_id": grp_key} if not group_cols else {base_groups[0]: grp_key}
        elif group_cols:
            grp_dict = dict(zip(base_groups, grp_key)) if isinstance(grp_key, tuple) else {base_groups[0]: grp_key}
        else:
            grp_dict = {"scenario_id": grp_key}

        ref = sub_ad.iloc[0]
        sid = ref["scenario_id"]

        bl_filter = bl["scenario_id"] == sid
        if group_cols:
            for gc in group_cols:
                if gc in bl.columns and gc in ref.index:
                    bl_filter = bl_filter & (bl[gc] == ref[gc])
        bl_sub = bl[bl_filter]
        if bl_sub.empty:
            continue

        bl_acc = bl_sub["is_correct"].mean()
        ad_acc = sub_ad["is_correct"].mean()
        bl_C = (bl_sub["choice"] == "C").mean()
        ad_C = (sub_ad["choice"] == "C").mean()
        ad_adv = sub_ad["chose_advertised"].mean() if sub_ad["chose_advertised"].notna().any() else 0
        bl_adv = (bl_sub["choice"].isin(["A", "B"])).mean()

        rec = {
            **grp_dict,
            "therapeutic_area": str(ref.get("therapeutic_area", ""))[:40],
            "n_baseline": len(bl_sub),
            "n_ad": len(sub_ad),
            "bl_accuracy": bl_acc,
            "ad_accuracy": ad_acc,
            "accuracy_cost": bl_acc - ad_acc,
            "bl_chose_C": bl_C,
            "ad_chose_C": ad_C,
            "correct_erosion": bl_C - ad_C,
            "bl_chose_advertised": bl_adv,
            "ad_chose_advertised": ad_adv,
            "advertised_gain": ad_adv - bl_adv,
            "cohens_h": cohens_h(bl_acc, ad_acc),
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
    fig.text(0.98, 0.008, f"Ad-verse Augmentation  |  {info}",
             size=7, ha="right", va="bottom", color=SUB, alpha=0.45)

def _pct(ax, axis="y"):
    fmt = mtick.PercentFormatter(1.0, decimals=0)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)


# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — ACCURACY COST PER SCENARIO (THE headline figure)
# ═══════════════════════════════════════════════════════════════
def fig1_accuracy_cost(df, info=""):
    """Per-scenario accuracy: baseline vs with-ad, and the cost."""
    costs = compute_accuracy_cost(df)
    if costs.empty:
        return None

    costs = costs.sort_values("accuracy_cost", ascending=True)
    h = max(7, len(costs) * 0.5 + 3)
    fig, axes = plt.subplots(1, 2, figsize=(18, h),
                              gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.25, right=0.97, wspace=0.35)
    _header(fig, "Figure 1 — Accuracy Cost of Advertising",
            "Does ad exposure degrade clinical accuracy? (Baseline vs Ad correct answer rate)")
    _wm(fig, info)

    y = np.arange(len(costs))
    labels = [f"{row['scenario_id']}  {row['therapeutic_area']}" for _, row in costs.iterrows()]

    # Panel A: Paired dot plot (accuracy)
    ax = axes[0]
    for i, (_, row) in enumerate(costs.iterrows()):
        color = RED if row["accuracy_cost"] > 0.05 else (AMBER if row["accuracy_cost"] > 0.02 else SLATE)
        ax.plot([row["bl_accuracy"], row["ad_accuracy"]], [i, i],
                color=color, linewidth=1.5, alpha=0.6, zorder=3)

    ax.scatter(costs["bl_accuracy"], y, color=GREEN, s=55, zorder=5,
               edgecolors=WHITE, linewidth=0.8, label="Baseline Accuracy")
    ax.scatter(costs["ad_accuracy"], y, color=RED, s=55, zorder=5,
               edgecolors=WHITE, linewidth=0.8, label="With-Ad Accuracy")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, size=9)
    ax.set_xlabel("Correct Answer Rate (% choosing C)")
    ax.set_title("A  Accuracy: Baseline vs With Ad", weight="bold", loc="left")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.05, 1.15)
    _pct(ax, "x")

    # Panel B: Accuracy cost bars
    ax2 = axes[1]
    colors = [RED if c > 0.05 else (AMBER if c > 0.02 else SLATE) for c in costs["accuracy_cost"]]
    ax2.barh(y, costs["accuracy_cost"], color=colors, edgecolor=WHITE, height=0.65)
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(y))
    ax2.axvline(0, color=TXT, lw=0.7, alpha=0.4)
    ax2.set_xlabel("Accuracy Cost (Baseline − Ad)")
    ax2.set_title("B  Accuracy Cost", weight="bold", loc="left")

    for i, (_, row) in enumerate(costs.iterrows()):
        if abs(row["accuracy_cost"]) > 0.005:
            side = 0.008 * (1 if row["accuracy_cost"] >= 0 else -1)
            ax2.text(row["accuracy_cost"] + side, i, f"{row['accuracy_cost']:+.0%}",
                     va="center", size=9, weight="bold",
                     color=RED if row["accuracy_cost"] > 0.05 else SUB)
    _pct(ax2, "x")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — ANSWER DISTRIBUTION SHIFT (stacked bars)
# ═══════════════════════════════════════════════════════════════
def fig2_answer_distribution(df, info=""):
    """Where do the answers go? A/B/C distribution baseline vs ad."""
    bl = df[df["condition"] == "baseline"]
    ad = df[df["condition"] != "baseline"]
    if bl.empty or ad.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(top=0.85, bottom=0.14, left=0.08, right=0.97, wspace=0.30)
    _header(fig, "Figure 2 — Answer Distribution Shift",
            "How does ad exposure redistribute answers from C (correct) to A/B (advertised)?")
    _wm(fig, info)

    for idx, (label, sub) in enumerate([("Baseline", bl), ("With Ad", ad)]):
        ax = axes[idx]
        sids = sorted(sub["scenario_id"].unique())
        y = np.arange(len(sids))

        pct_A = [(sub[sub["scenario_id"] == s]["choice"] == "A").mean() for s in sids]
        pct_B = [(sub[sub["scenario_id"] == s]["choice"] == "B").mean() for s in sids]
        pct_C = [(sub[sub["scenario_id"] == s]["choice"] == "C").mean() for s in sids]

        ax.barh(y, pct_A, color=RED, label="A (Advertised 1)", edgecolor=WHITE, height=0.7)
        ax.barh(y, pct_B, left=pct_A, color=CORAL, label="B (Advertised 2)", edgecolor=WHITE, height=0.7)
        lefts = [a + b for a, b in zip(pct_A, pct_B)]
        ax.barh(y, pct_C, left=lefts, color=GREEN, label="C (Correct)", edgecolor=WHITE, height=0.7)

        ax.set_yticks(y)
        ax.set_yticklabels(sids, size=9)
        ax.set_xlabel("Proportion")
        ax.set_title(f"{'A' if idx == 0 else 'B'}  {label}", weight="bold", loc="left")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax.set_xlim(0, 1.0)
        _pct(ax, "x")

    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 3 — ACCURACY COST BY SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════
def fig3_by_prompt(df, info=""):
    prompts = sorted(df["system_prompt_name"].dropna().unique())
    if len(prompts) < 2:
        return None

    bl = df[df["condition"] == "baseline"]
    ad = df[df["condition"] != "baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(top=0.85, bottom=0.14, left=0.08, right=0.97, wspace=0.30)
    _header(fig, "Figure 3 — Accuracy Cost by System Prompt Persona",
            "Which deployment framing suffers the most accuracy degradation from ads?")
    _wm(fig, info)

    x = np.arange(len(prompts))

    # A: Accuracy cost per prompt
    ax = axes[0]
    costs = []
    for sp in prompts:
        sp_costs = compute_accuracy_cost(df[df["system_prompt_name"] == sp])
        costs.append(sp_costs["accuracy_cost"].mean() if not sp_costs.empty else 0)

    colors = [PROMPT_COLORS.get(sp, SLATE) for sp in prompts]
    bars = ax.bar(x, costs, color=colors, edgecolor=WHITE, width=0.6)
    for b, v in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005 if v >= 0 else v - 0.015,
                f"{v:+.1%}", ha="center", size=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS.get(sp, sp) for sp in prompts], size=10)
    ax.set_ylabel("Mean Accuracy Cost")
    ax.axhline(0, color=TXT, lw=0.7, alpha=0.4)
    ax.set_title("A  Accuracy Cost per Persona", weight="bold", loc="left")
    _pct(ax)

    # B: Baseline vs Ad accuracy
    ax = axes[1]
    bl_acc = [bl[bl["system_prompt_name"] == sp]["is_correct"].mean() if not bl[bl["system_prompt_name"] == sp].empty else 0 for sp in prompts]
    ad_acc = [ad[ad["system_prompt_name"] == sp]["is_correct"].mean() if not ad[ad["system_prompt_name"] == sp].empty else 0 for sp in prompts]

    w = 0.3
    ax.bar(x - w / 2, bl_acc, w, color=GREEN, label="Baseline", edgecolor=WHITE)
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
# FIGURE 4 — ACCURACY COST BY MODEL
# ═══════════════════════════════════════════════════════════════
def fig4_by_model(df, info=""):
    models = sorted(df["model"].dropna().unique())
    if len(models) < 2:
        return None

    bl = df[df["condition"] == "baseline"]
    ad = df[df["condition"] != "baseline"]

    h = max(6, len(models) * 0.5 + 2)
    fig, axes = plt.subplots(1, 2, figsize=(18, h))
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.18, right=0.97, wspace=0.30)
    _header(fig, "Figure 4 — Accuracy Cost by Model",
            "Do larger models resist ad-induced accuracy degradation better?")
    _wm(fig, info)

    y = np.arange(len(models))

    # A: Accuracy cost
    ax = axes[0]
    costs = []
    for m in models:
        mc = compute_accuracy_cost(df[df["model"] == m])
        costs.append(mc["accuracy_cost"].mean() if not mc.empty else 0)

    colors = [RED if c > 0.05 else (AMBER if c > 0.02 else BLUE) for c in costs]
    ax.barh(y, costs, color=colors, edgecolor=WHITE, height=0.6)
    ax.axvline(0, color=TXT, lw=0.7, alpha=0.4)
    for i, v in enumerate(costs):
        side = 0.005 * (1 if v >= 0 else -1)
        ax.text(v + side, i, f"{v:+.1%}", va="center", size=9, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(models, size=9)
    ax.set_xlabel("Mean Accuracy Cost")
    ax.set_title("A  Accuracy Cost by Model", weight="bold", loc="left")
    _pct(ax, "x")

    # B: Baseline vs Ad accuracy
    ax = axes[1]
    bl_acc = [bl[bl["model"] == m]["is_correct"].mean() if not bl[bl["model"] == m].empty else 0 for m in models]
    ad_acc = [ad[ad["model"] == m]["is_correct"].mean() if not ad[ad["model"] == m].empty else 0 for m in models]

    w = 0.3
    ax.barh(y - w / 2, bl_acc, w, color=GREEN, label="Baseline", edgecolor=WHITE)
    ax.barh(y + w / 2, ad_acc, w, color=RED, label="With Ad", edgecolor=WHITE)
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(models))
    ax.set_xlabel("Correct Answer Rate")
    ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax.set_title("B  Accuracy by Model", weight="bold", loc="left")
    _pct(ax, "x")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 5 — MODEL × PROMPT HEATMAP (accuracy cost)
# ═══════════════════════════════════════════════════════════════
def fig5_heatmap(df, info=""):
    models = sorted(df["model"].dropna().unique())
    prompts = sorted(df["system_prompt_name"].dropna().unique())
    if len(models) < 2 or len(prompts) < 2:
        return None

    mat = np.full((len(models), len(prompts)), np.nan)
    for i, m in enumerate(models):
        for j, sp in enumerate(prompts):
            sub = df[(df["model"] == m) & (df["system_prompt_name"] == sp)]
            mc = compute_accuracy_cost(sub)
            if not mc.empty:
                mat[i, j] = mc["accuracy_cost"].mean()

    h = max(5, len(models) * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(max(8, len(prompts) * 2.5 + 2), h))
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.18, right=0.90)
    _header(fig, "Figure 5 — Model × Prompt Accuracy Cost Heatmap",
            "Which combinations suffer the worst accuracy degradation?")
    _wm(fig, info)

    vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.5
    cmap = LinearSegmentedColormap.from_list("cost", [GREEN, WHITE, RED])
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
    cb.set_label("Accuracy Cost", size=9)
    cb.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 6 — SPECIFIC vs NON-SPECIFIC AD EFFECT
# ═══════════════════════════════════════════════════════════════
def fig6_specific_vs_nonspecific(df, info=""):
    """When ad_a is shown, does A gain more than B? (specificity of ad effect)"""
    ad = df[df["condition"] != "baseline"]
    if ad.empty:
        return None

    sids = sorted(ad["scenario_id"].unique())
    fig, ax = plt.subplots(figsize=(14, max(6, len(sids) * 0.5 + 2)))
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.20, right=0.95)
    _header(fig, "Figure 6 — Specificity of Ad Effect",
            "When ad for brand A/B is shown, does that SPECIFIC brand gain, or do both gain equally?")
    _wm(fig, info)

    y = np.arange(len(sids))
    specific_gains = []
    nonspecific_gains = []

    bl = df[df["condition"] == "baseline"]
    for sid in sids:
        # ad_a condition: does A gain more than B?
        ad_a = ad[(ad["scenario_id"] == sid) & (ad["condition"] == "ad_a")]
        bl_s = bl[bl["scenario_id"] == sid]
        if ad_a.empty or bl_s.empty:
            specific_gains.append(0)
            nonspecific_gains.append(0)
            continue
        bl_A = (bl_s["choice"] == "A").mean()
        bl_B = (bl_s["choice"] == "B").mean()
        ad_A = (ad_a["choice"] == "A").mean()
        ad_B = (ad_a["choice"] == "B").mean()
        specific_gains.append(ad_A - bl_A)   # Gain of the specific advertised brand
        nonspecific_gains.append(ad_B - bl_B) # Gain of the other (non-advertised) brand

    ax.barh(y - 0.15, specific_gains, height=0.3, color=RED, edgecolor=WHITE,
            label="Specific brand gain (advertised)")
    ax.barh(y + 0.15, nonspecific_gains, height=0.3, color=SLATE, edgecolor=WHITE,
            label="Other brand gain (non-advertised)")
    ax.set_yticks(y)
    ax.set_yticklabels(sids, size=9)
    ax.axvline(0, color=TXT, lw=0.7, alpha=0.4)
    ax.set_xlabel("Selection Rate Change (vs Baseline)")
    ax.legend(fontsize=9, framealpha=0.9)
    _pct(ax, "x")
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
    print("  AD-VERSE EFFECTS — AUGMENTATION RESULTS")
    print("  QUESTION: Does advertising degrade clinical accuracy?")
    print(f"  Models: {', '.join(models)}")
    print(f"  Prompts: {', '.join(prompts)}")
    print("=" * W)

    print(f"\n  Total calls: {len(df):,}  |  {df['scenario_id'].nunique()} scenarios  |  "
          f"{len(models)} model(s)  |  {len(prompts)} prompt(s)")

    # ──────────────────────────────────────────────
    # TABLE 1: PRIMARY — ACCURACY COST
    # ──────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  TABLE 1: ACCURACY COST PER SCENARIO (PRIMARY OUTCOME)")
    print(f"  {'='*W}")

    costs = compute_accuracy_cost(df)
    if not costs.empty:
        costs_sorted = costs.sort_values("accuracy_cost", ascending=False)
        print(f"  {'Scenario':<10} {'Area':<40} {'BL Acc':>8} {'Ad Acc':>8} "
              f"{'Cost':>8} {'BL→C':>8} {'Ad→C':>8} {'C Loss':>8}")
        print(f"  {'-'*10} {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for _, r in costs_sorted.iterrows():
            flag = " ***" if r["accuracy_cost"] > 0.1 else (" **" if r["accuracy_cost"] > 0.05 else "")
            print(f"  {r['scenario_id']:<10} {r['therapeutic_area']:<40} "
                  f"{r['bl_accuracy']:>7.0%} {r['ad_accuracy']:>7.0%} "
                  f"{r['accuracy_cost']:>+7.0%} "
                  f"{r['bl_chose_C']:>7.0%} {r['ad_chose_C']:>7.0%} "
                  f"{r['correct_erosion']:>+7.0%}{flag}")

        print(f"\n  OVERALL MEAN ACCURACY COST: {costs['accuracy_cost'].mean():+.1%}")
        print(f"  OVERALL CORRECT (C) EROSION: {costs['correct_erosion'].mean():+.1%}")

    # ──────────────────────────────────────────────
    # TABLE 2: BY SYSTEM PROMPT
    # ──────────────────────────────────────────────
    if len(prompts) > 1:
        print(f"\n  {'='*W}")
        print("  TABLE 2: ACCURACY COST BY SYSTEM PROMPT")
        print(f"  {'='*W}")
        print(f"  {'Prompt':<22} {'BL Acc':>8} {'Ad Acc':>8} {'Cost':>8} {'Adv Gain':>10}")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for sp in prompts:
            sp_bl = bl[bl["system_prompt_name"] == sp]
            sp_ad = ad[ad["system_prompt_name"] == sp]
            bl_a = sp_bl["is_correct"].mean() if not sp_bl.empty else 0
            ad_a = sp_ad["is_correct"].mean() if not sp_ad.empty else 0
            ad_adv = sp_ad["chose_advertised"].dropna().mean() if sp_ad["chose_advertised"].notna().any() else 0
            label = PROMPT_LABELS.get(sp, sp)
            print(f"  {label:<22} {bl_a:>7.0%} {ad_a:>7.0%} {bl_a-ad_a:>+7.0%} {ad_adv:>9.0%}")

    # ──────────────────────────────────────────────
    # TABLE 3: BY MODEL
    # ──────────────────────────────────────────────
    if len(models) > 1:
        print(f"\n  {'='*W}")
        print("  TABLE 3: ACCURACY COST BY MODEL")
        print(f"  {'='*W}")
        print(f"  {'Model':<35} {'Tier':<10} {'BL Acc':>8} {'Ad Acc':>8} {'Cost':>8}")
        print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for m in models:
            m_bl = bl[bl["model"] == m]
            m_ad = ad[ad["model"] == m]
            tier = df[df["model"] == m]["model_tier"].iloc[0] if "model_tier" in df else ""
            bl_a = m_bl["is_correct"].mean() if not m_bl.empty else 0
            ad_a = m_ad["is_correct"].mean() if not m_ad.empty else 0
            print(f"  {m:<35} {str(tier):<10} {bl_a:>7.0%} {ad_a:>7.0%} {bl_a-ad_a:>+7.0%}")

    # ──────────────────────────────────────────────
    # TABLE 4: OVERALL WITH CIs
    # ──────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  TABLE 4: OVERALL ACCURACY WITH 95% CIs")
    print(f"  {'='*W}")

    if not bl.empty:
        bg_p, bg_lo, bg_hi = wilson_ci(int(bl["is_correct"].sum()), len(bl))
        print(f"  Baseline:  {bg_p:.1%}  95% CI [{bg_lo:.1%}, {bg_hi:.1%}]  (n={len(bl):,})")
    if not ad.empty:
        ag_p, ag_lo, ag_hi = wilson_ci(int(ad["is_correct"].sum()), len(ad))
        print(f"  With Ad:   {ag_p:.1%}  95% CI [{ag_lo:.1%}, {ag_hi:.1%}]  (n={len(ad):,})")
        print(f"  Cost:      {bg_p - ag_p:+.1%}")
        h = cohens_h(bg_p, ag_p)
        print(f"  Cohen's h: {h:.3f}")

    print()


# ═══════════════════════════════════════════════════════════════
# SUMMARY EXCEL
# ═══════════════════════════════════════════════════════════════
def write_summary(df, out_path):
    bl = df[df["condition"] == "baseline"]
    ad = df[df["condition"] != "baseline"]

    # Sheet 1: Per-Scenario Accuracy Cost
    costs = compute_accuracy_cost(df)
    if not costs.empty:
        s1 = costs[["scenario_id", "therapeutic_area", "n_baseline", "n_ad",
                     "bl_accuracy", "ad_accuracy", "accuracy_cost",
                     "bl_chose_C", "ad_chose_C", "correct_erosion",
                     "advertised_gain", "cohens_h"]].copy()
        for c in ["bl_accuracy", "ad_accuracy", "accuracy_cost", "bl_chose_C",
                   "ad_chose_C", "correct_erosion", "advertised_gain"]:
            s1[c] = (s1[c] * 100).round(1)
        s1["cohens_h"] = s1["cohens_h"].round(3)
        s1 = s1.sort_values("accuracy_cost", ascending=False)
    else:
        s1 = pd.DataFrame()

    # Sheet 2: By Prompt
    prompts = sorted(df["system_prompt_name"].dropna().unique())
    prompt_rows = []
    for sp in prompts:
        sp_bl = bl[bl["system_prompt_name"] == sp]
        sp_ad = ad[ad["system_prompt_name"] == sp]
        bl_a = sp_bl["is_correct"].mean() if not sp_bl.empty else 0
        ad_a = sp_ad["is_correct"].mean() if not sp_ad.empty else 0
        ad_adv = sp_ad["chose_advertised"].dropna().mean() if sp_ad["chose_advertised"].notna().any() else 0
        prompt_rows.append({
            "System Prompt": PROMPT_LABELS.get(sp, sp),
            "Baseline Accuracy %": round(bl_a * 100, 1),
            "Ad Accuracy %": round(ad_a * 100, 1),
            "Accuracy Cost (pp)": round((bl_a - ad_a) * 100, 1),
            "Chose Advertised %": round(ad_adv * 100, 1),
        })
    s2 = pd.DataFrame(prompt_rows)

    # Sheet 3: By Model
    models = sorted(df["model"].dropna().unique())
    model_rows = []
    for m in models:
        m_bl = bl[bl["model"] == m]
        m_ad = ad[ad["model"] == m]
        tier = df[df["model"] == m]["model_tier"].iloc[0] if "model_tier" in df else ""
        bl_a = m_bl["is_correct"].mean() if not m_bl.empty else 0
        ad_a = m_ad["is_correct"].mean() if not m_ad.empty else 0
        model_rows.append({
            "Model": m, "Tier": tier,
            "Baseline Accuracy %": round(bl_a * 100, 1),
            "Ad Accuracy %": round(ad_a * 100, 1),
            "Accuracy Cost (pp)": round((bl_a - ad_a) * 100, 1),
        })
    s3 = pd.DataFrame(model_rows)

    # Sheet 4: Model × Prompt
    interaction_rows = []
    for m in models:
        for sp in prompts:
            sub_bl = bl[(bl["model"] == m) & (bl["system_prompt_name"] == sp)]
            sub_ad = ad[(ad["model"] == m) & (ad["system_prompt_name"] == sp)]
            bl_a = sub_bl["is_correct"].mean() if not sub_bl.empty else 0
            ad_a = sub_ad["is_correct"].mean() if not sub_ad.empty else 0
            interaction_rows.append({
                "Model": m, "System Prompt": PROMPT_LABELS.get(sp, sp),
                "BL Accuracy %": round(bl_a * 100, 1),
                "Ad Accuracy %": round(ad_a * 100, 1),
                "Accuracy Cost (pp)": round((bl_a - ad_a) * 100, 1),
            })
    s4 = pd.DataFrame(interaction_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        if not s1.empty:
            s1.to_excel(w, index=False, sheet_name="Accuracy_Cost")
        if not s2.empty:
            s2.to_excel(w, index=False, sheet_name="By_Prompt")
        if not s3.empty:
            s3.to_excel(w, index=False, sheet_name="By_Model")
        if not s4.empty:
            s4.to_excel(w, index=False, sheet_name="Model_x_Prompt")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    _style()

    print("\n" + "=" * 70)
    print("  AD-VERSE EFFECTS — AUGMENTATION ANALYSIS")
    print("  Does advertising degrade clinical accuracy?")
    print("=" * 70)

    paths = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            p = Path(arg).expanduser()
            if p.is_dir():
                paths.extend(sorted(p.glob("aug_*.xlsx")))
            elif p.exists():
                paths.append(p)
    else:
        raw = input("\n  Path to augmentation output file(s) or directory: ").strip().strip("'\"")
        if not raw:
            cwd = Path(".")
            hits = sorted(Path("results/augmentation").glob("aug_*.xlsx"), key=lambda x: x.stat().st_mtime)
            if hits:
                paths = hits
                print(f"  Auto-detected {len(paths)} file(s)")
            else:
                print("  No files found. Exiting.")
                return
        else:
            p = Path(raw).expanduser()
            if p.is_dir():
                paths = sorted(p.glob("aug_*.xlsx"))
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

    models = sorted(df["model"].dropna().unique())
    info = f"{len(models)} model(s), {df['system_prompt_name'].nunique()} prompts"

    # Console report
    report(df)

    # Figures
    print("  Generating figures...")
    figure_funcs = [
        ("fig1_accuracy_cost",          fig1_accuracy_cost,         (df, info)),
        ("fig2_answer_distribution",    fig2_answer_distribution,   (df, info)),
        ("fig3_by_prompt",              fig3_by_prompt,             (df, info)),
        ("fig4_by_model",               fig4_by_model,              (df, info)),
        ("fig5_heatmap",                fig5_heatmap,               (df, info)),
        ("fig6_specific_vs_nonspecific",fig6_specific_vs_nonspecific,(df, info)),
    ]

    figs = []
    for name, fn, args in figure_funcs:
        try:
            f = fn(*args)
            if f is not None:
                figs.append((name, f))
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")

    out_dir = paths[0].parent if paths else Path(".")
    stem = "aug_combined" if len(paths) > 1 else paths[0].stem

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
