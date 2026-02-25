#!/usr/bin/env python3
"""
Ad-verse Effects — Publication Figure Generation
==================================================
Generates main and supplementary figures from raw result files.

Main figures:
  Figure 1: Two-panel (A) preference shift by model, (B) accuracy by model
  Figure 2: Provider-level comparison with CIs

Supplementary figures:
  Figure S1: Heatmap of shift by model × scenario
  Figure S2: Wellness endorsement by model
  Figure S3: Augmentation accuracy by model
  Figure S4: Preference shift by system prompt persona

Usage:
    python generate_figures.py [results_main_dir] [results_aug_dir]

    If no arguments given, reads from results/main/ and results/augmentation/.
    Raw result Excel files (adverse_*.xlsx, aug_*.xlsx) must be present.

Output:
    figures/main/        Figure_1.png, Figure_1.pdf, Figure_2.png, Figure_2.pdf
    figures/supplementary/  Figure_S1–S4 (.png, .pdf)
"""

import sys
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 0.75,
})

# Provider colors
C_OPENAI = "#4A7FB5"
C_ANTHRO = "#D4756B"
C_GOOGLE = "#6BAD70"
C_OVERALL = "#555555"

PROVIDER_COLORS = {
    "OpenAI": C_OPENAI,
    "Anthropic": C_ANTHRO,
    "Google": C_GOOGLE,
}


def _color_for_model(model_id: str) -> str:
    if "gpt" in model_id or "o4" in model_id:
        return C_OPENAI
    elif "claude" in model_id:
        return C_ANTHRO
    elif "gemini" in model_id:
        return C_GOOGLE
    return C_OVERALL


def _short_name(model_id: str) -> str:
    replacements = [
        ("gpt-4.1-mini", "GPT-4.1\nMini"),
        ("gpt-4.1", "GPT-4.1"),
        ("gpt-5-mini", "GPT-5\nMini"),
        ("gpt-5.2", "GPT-5.2"),
        ("o4-mini", "o4-mini"),
        ("claude-haiku-4-5", "Claude\nHaiku 4.5"),
        ("claude-sonnet-4-5", "Claude\nSonnet 4.5"),
        ("claude-sonnet-4-6", "Claude\nSonnet 4.6"),
        ("claude-opus-4-6", "Claude\nOpus 4.6"),
        ("gemini-2.5-flash-lite", "Gemini 2.5\nFlash Lite"),
        ("gemini-2.5-flash", "Gemini 2.5\nFlash"),
        ("gemini-3-flash-preview", "Gemini 3\nFlash Preview"),
    ]
    for old, new in replacements:
        if model_id == old:
            return new
    return model_id


def load_results(main_dir: str, aug_dir: str):
    """Load all raw result Excel files."""
    main_files = sorted(glob.glob(os.path.join(main_dir, "adverse_*.xlsx")))
    aug_files = sorted(glob.glob(os.path.join(aug_dir, "aug_*.xlsx")))

    main_dfs = [pd.read_excel(f) for f in main_files]
    aug_dfs = [pd.read_excel(f) for f in aug_files]

    main = pd.concat(main_dfs, ignore_index=True) if main_dfs else pd.DataFrame()
    aug = pd.concat(aug_dfs, ignore_index=True) if aug_dfs else pd.DataFrame()

    return main, aug


def compute_rx_shifts(df: pd.DataFrame):
    """Compute per-model preference shift for Rx scenarios (S01–S13)."""
    rx = df[df["is_rx"] == True].copy()
    results = []
    for model, g in rx.groupby("model"):
        bl = g[g["condition"] == "baseline"]
        ad = g[g["condition"].isin(["ad_a", "ad_b"])]
        if len(bl) == 0 or len(ad) == 0:
            continue
        bl_adv_rate = ((bl["choice"] == "A").sum() + (bl["choice"] == "B").sum()) / (2 * len(bl))
        ad_adv_rate = ad["chose_advertised"].mean()
        shift = (ad_adv_rate - bl_adv_rate) * 100

        bl_acc = bl["is_correct"].mean() * 100
        ad_acc = ad["is_correct"].mean() * 100

        results.append({
            "model": model,
            "shift": shift,
            "bl_acc": bl_acc,
            "ad_acc": ad_acc,
        })
    return pd.DataFrame(results).sort_values("shift", ascending=False)


def compute_provider_shifts(df: pd.DataFrame):
    """Compute provider-level mean shifts."""
    rx = df[df["is_rx"] == True].copy()
    results = []
    for provider, g in rx.groupby("provider"):
        bl = g[g["condition"] == "baseline"]
        ad = g[g["condition"].isin(["ad_a", "ad_b"])]
        if len(bl) == 0 or len(ad) == 0:
            continue
        bl_adv_rate = ((bl["choice"] == "A").sum() + (bl["choice"] == "B").sum()) / (2 * len(bl))
        ad_adv_rate = ad["chose_advertised"].mean()
        shift = (ad_adv_rate - bl_adv_rate) * 100
        results.append({"provider": provider, "shift": shift})
    return pd.DataFrame(results)


def figure_1(rx_data: pd.DataFrame, outdir: str):
    """Figure 1: Two-panel — (A) shift by model, (B) accuracy by model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    models = rx_data["model"].values
    shifts = rx_data["shift"].values
    colors = [_color_for_model(m) for m in models]
    names = [_short_name(m) for m in models]

    # Panel A
    ax1.bar(range(len(models)), shifts, color=colors, width=0.7,
            edgecolor="white", linewidth=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.4)
    mean_shift = shifts.mean()
    ax1.axhline(y=mean_shift, color="#888888", linewidth=0.5, linestyle="--", alpha=0.7)
    ax1.text(len(models)-0.5, mean_shift + 1.2, f"Mean +{mean_shift:.1f}",
             fontsize=6, color="#666666", ha="right")
    ax1.set_ylabel("Preference shift (pp)")
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=5.5)
    ax1.set_ylim(min(shifts) - 5, max(shifts) + 8)
    ax1.set_title("A", fontsize=10, fontweight="bold", loc="left", x=-0.08)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    legend_elements = [Patch(facecolor=C_GOOGLE, label="Google"),
                       Patch(facecolor=C_OPENAI, label="OpenAI"),
                       Patch(facecolor=C_ANTHRO, label="Anthropic")]
    ax1.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=6)

    # Panel B
    bl_acc = rx_data["bl_acc"].values
    ad_acc = rx_data["ad_acc"].values
    x = np.arange(len(models))
    w = 0.35
    ax2.bar(x - w/2, bl_acc, w, color="#CCCCCC", label="Baseline",
            edgecolor="white", linewidth=0.3)
    ax2.bar(x + w/2, ad_acc, w, color=colors, label="With ad",
            edgecolor="white", linewidth=0.3, alpha=0.8)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=5.5)
    ax2.set_ylim(65, 102)
    ax2.set_title("B", fontsize=10, fontweight="bold", loc="left", x=-0.08)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(["Baseline", "With ad"], loc="lower right", frameon=False, fontsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure_1.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(outdir, "Figure_1.pdf"), bbox_inches="tight")
    plt.close()
    print("  Figure 1 done")


def figure_2(prov_data: pd.DataFrame, outdir: str):
    """Figure 2: Provider-level comparison."""
    prov_data = prov_data.sort_values("shift", ascending=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    providers = prov_data["provider"].values
    shifts = prov_data["shift"].values
    colors = [PROVIDER_COLORS.get(p, C_OVERALL) for p in providers]

    bars = ax.bar(range(len(providers)), shifts, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.3)
    ax.set_ylabel("Mean preference shift (pp)")
    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels(providers)
    ax.set_ylim(0, max(shifts) + 8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(shifts):
        ax.text(i, v + 1.0, f"+{v:.1f}", ha="center", fontsize=7, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure_2_Provider.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(outdir, "Figure_2_Provider.pdf"), bbox_inches="tight")
    plt.close()
    print("  Figure 2 done")


def figure_s1_heatmap(df: pd.DataFrame, outdir: str):
    """Figure S1: Model × scenario shift heatmap."""
    rx = df[df["is_rx"] == True].copy()
    pivot_data = []
    for (model, scenario), g in rx.groupby(["model", "scenario_id"]):
        bl = g[g["condition"] == "baseline"]
        ad = g[g["condition"].isin(["ad_a", "ad_b"])]
        if len(bl) == 0 or len(ad) == 0:
            continue
        bl_a = (bl["choice"] == "A").mean()
        bl_b = (bl["choice"] == "B").mean()
        proxy = (bl_a + bl_b) / 2
        chose_adv = ad["chose_advertised"].mean()
        shift = (chose_adv - proxy) * 100
        pivot_data.append({"model": model, "scenario": scenario, "shift": shift})

    pdf = pd.DataFrame(pivot_data)
    heatmap = pdf.pivot(index="model", columns="scenario", values="shift")
    model_order = heatmap.mean(axis=1).sort_values(ascending=False).index
    scenario_order = sorted(heatmap.columns)
    heatmap = heatmap.loc[model_order, scenario_order]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(heatmap.values, cmap="RdBu_r", aspect="auto", vmin=-20, vmax=60)
    ax.set_xticks(range(len(scenario_order)))
    ax.set_xticklabels(scenario_order, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(model_order)))

    short = [_short_name(m).replace("\n", " ") for m in model_order]
    ax.set_yticklabels(short, fontsize=6)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Shift (pp)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Model")

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure_S1_Heatmap.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(outdir, "Figure_S1_Heatmap.pdf"), bbox_inches="tight")
    plt.close()
    print("  Figure S1 (heatmap) done")


def figure_s2_wellness(df: pd.DataFrame, outdir: str):
    """Figure S2: Wellness endorsement by model."""
    well = df[df["is_rx"] == False].copy()
    results = []
    for model, g in well.groupby("model"):
        bl = g[g["condition"] == "baseline"]
        ad = g[g["condition"] == "ad"]
        bl_rate = (bl["choice"] == "A").mean() * 100 if len(bl) > 0 else 0
        ad_rate = (ad["choice"] == "A").mean() * 100 if len(ad) > 0 else 0
        results.append({"model": model, "bl": bl_rate, "ad": ad_rate})

    rdf = pd.DataFrame(results).sort_values("model")
    models = rdf["model"].values
    names = [_short_name(m) for m in models]
    colors = [_color_for_model(m) for m in models]

    fig, ax = plt.subplots(figsize=(5, 2.5))
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, rdf["bl"].values, w, color="#CCCCCC", label="Baseline",
           edgecolor="white", linewidth=0.3)
    ax.bar(x + w/2, rdf["ad"].values, w, color=colors, alpha=0.7, label="With ad",
           edgecolor="white", linewidth=0.3)
    ax.set_ylabel("Endorsement rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=5.5)
    ax.set_ylim(0, max(max(rdf["bl"]), max(rdf["ad"])) + 4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure_S2_Wellness.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(outdir, "Figure_S2_Wellness.pdf"), bbox_inches="tight")
    plt.close()
    print("  Figure S2 (wellness) done")


def figure_s3_augmentation(aug: pd.DataFrame, outdir: str):
    """Figure S3: Augmentation accuracy by model."""
    results = []
    for model, g in aug.groupby("model"):
        bl = g[g["condition"] == "baseline"]
        ad = g[g["condition"].isin(["ad_a", "ad_b"])]
        bl_acc = bl["is_correct"].mean() * 100 if len(bl) > 0 else 0
        ad_acc = ad["is_correct"].mean() * 100 if len(ad) > 0 else 0
        results.append({"model": model, "bl_acc": bl_acc, "ad_acc": ad_acc})

    rdf = pd.DataFrame(results).sort_values("model")
    models = rdf["model"].values
    names = [_short_name(m) for m in models]
    colors = [_color_for_model(m) for m in models]

    fig, ax = plt.subplots(figsize=(5, 2.5))
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, rdf["bl_acc"].values, w, color="#CCCCCC", label="Baseline",
           edgecolor="white", linewidth=0.3)
    ax.bar(x + w/2, rdf["ad_acc"].values, w, color=colors, alpha=0.8, label="With ad",
           edgecolor="white", linewidth=0.3)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=5.5)
    ax.set_ylim(55, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="lower left")

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure_S3_Augmentation.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(outdir, "Figure_S3_Augmentation.pdf"), bbox_inches="tight")
    plt.close()
    print("  Figure S3 (augmentation) done")


def figure_s4_persona(df: pd.DataFrame, outdir: str):
    """Figure S4: Preference shift by system prompt persona."""
    rx = df[df["is_rx"] == True].copy()
    results = []
    for prompt, g in rx.groupby("system_prompt_name"):
        bl = g[g["condition"] == "baseline"]
        ad = g[g["condition"].isin(["ad_a", "ad_b"])]
        if len(bl) == 0 or len(ad) == 0:
            continue
        bl_adv_rate = ((bl["choice"] == "A").sum() + (bl["choice"] == "B").sum()) / (2 * len(bl))
        ad_adv_rate = ad["chose_advertised"].mean()
        shift = (ad_adv_rate - bl_adv_rate) * 100
        results.append({"persona": prompt, "shift": shift})

    rdf = pd.DataFrame(results).sort_values("shift", ascending=True)
    labels = {
        "physician": "Physician",
        "helpful_ai": "Helpful AI",
        "customer_service": "Customer\nservice",
        "no_persona": "No persona",
    }

    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    personas = rdf["persona"].values
    shifts = rdf["shift"].values
    names = [labels.get(p, p) for p in personas]

    ax.barh(range(len(personas)), shifts, color="#7A9CC6", height=0.5,
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Preference shift (pp)")
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, max(shifts) + 4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(shifts):
        ax.text(v + 0.3, i, f"+{v:.1f}", va="center", fontsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure_S4_Persona.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(outdir, "Figure_S4_Persona.pdf"), bbox_inches="tight")
    plt.close()
    print("  Figure S4 (persona) done")


def main():
    # Parse directories from CLI or use defaults
    base = os.path.dirname(os.path.abspath(__file__))
    main_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "results", "main")
    aug_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, "results", "augmentation")

    # Output directories
    main_fig_dir = os.path.join(base, "figures", "main")
    supp_fig_dir = os.path.join(base, "figures", "supplementary")
    os.makedirs(main_fig_dir, exist_ok=True)
    os.makedirs(supp_fig_dir, exist_ok=True)

    print(f"Loading results from:\n  Main: {main_dir}\n  Aug:  {aug_dir}")
    main_df, aug_df = load_results(main_dir, aug_dir)

    if main_df.empty:
        print("No main results found. Place adverse_*.xlsx files in results/main/")
        return

    print(f"\nLoaded {len(main_df):,} main rows, {len(aug_df):,} augmentation rows")
    print(f"Models: {main_df['model'].nunique()}")

    # Compute summaries
    rx_data = compute_rx_shifts(main_df)
    prov_data = compute_provider_shifts(main_df)

    # Generate main figures
    print("\nGenerating main figures:")
    figure_1(rx_data, main_fig_dir)
    figure_2(prov_data, main_fig_dir)

    # Generate supplementary figures
    print("\nGenerating supplementary figures:")
    figure_s1_heatmap(main_df, supp_fig_dir)
    figure_s2_wellness(main_df, supp_fig_dir)
    if not aug_df.empty:
        figure_s3_augmentation(aug_df, supp_fig_dir)
    figure_s4_persona(main_df, supp_fig_dir)

    print(f"\nAll figures saved to:\n  {main_fig_dir}\n  {supp_fig_dir}")


if __name__ == "__main__":
    main()
