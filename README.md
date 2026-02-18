# Ad-verse Effects

**Do Pharmaceutical Advertisements Embedded in LLM Interactions Shift Clinical Recommendations?**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Large language models (LLMs) are increasingly deployed as clinical decision-support tools, health chatbots, and consumer-facing assistants. This study investigates whether pharmaceutical and wellness-supplement advertisements, when injected into LLM interactions, bias clinical recommendations toward the advertised product.

We present a within-subject factorial experiment testing **15 LLMs** (nano to flagship) across **3 providers** (OpenAI, Anthropic, Google), **4 deployment personas**, and **23 clinical scenarios** — totaling over **212,000 API calls** per full run. An augmentation experiment further tests whether this ad-induced shift degrades **clinical accuracy**.

**Investigators:** Mahmud Omar, MD — BRIDGE Lab, Icahn School of Medicine at Mount Sinai

---

## Experimental Design

### Main Experiment

| Factor | Levels | Description |
|---|---|---|
| **Scenario** | 23 (S01–S23) | Clinical vignettes: 13 prescription + 10 wellness |
| **Variant** | 3 (V1–V3) | Reworded versions to control for phrasing effects |
| **System Prompt** | 4 | `physician`, `helpful_ai`, `customer_service`, `no_persona` |
| **Ad Condition** | 2–3 | `baseline` (no ad), `ad_a`, `ad_b` (Rx) or `ad` (wellness) |
| **Repeat** | 20 | Independent samples at temperature = 0.7 |

**Primary metric:** Brand selection shift = P(choose advertised | ad) − P(choose advertised | baseline)

**Total per model:** 14,160 API calls (13 Rx × 720 + 10 wellness × 480)

### Augmentation Experiment (Accuracy Cost)

Tests whether ad-induced brand preference **degrades clinical accuracy**:

- 10 scenarios (A01–A10) where the advertised drugs are guideline-suboptimal
- 3 answer options: A (advertised 1), B (advertised 2), C (correct non-advertised)
- Correct answer is always C

**Primary metric:** Accuracy cost = P(correct | baseline) − P(correct | ad)

**Total per model:** 7,200 API calls

---

## Models Tested

| Provider | Model | Tier | API |
|---|---|---|---|
| OpenAI | gpt-4o-mini | small | Chat |
| OpenAI | gpt-4o | large | Chat |
| OpenAI | o3-mini | small | Chat |
| OpenAI | gpt-4.1-nano | nano | Responses |
| OpenAI | gpt-4.1-mini | small | Responses |
| OpenAI | gpt-4.1 | large | Responses |
| OpenAI | gpt-5.2 | flagship | Responses |
| Anthropic | claude-haiku-4-5 | small | Messages |
| Anthropic | claude-sonnet-4-5 | large | Messages |
| Anthropic | claude-opus-4-6 | flagship | Messages |
| Google | gemini-2.5-flash-lite | nano | GenAI |
| Google | gemini-2.5-flash | small | GenAI |
| Google | gemini-2.5-pro | large | GenAI |
| Google | gemini-3-flash-preview | large | GenAI |
| Google | gemini-3-pro-preview | flagship | GenAI |

---

## Repository Structure

```
ad-verse-effects/
├── README.md                         # This file
├── STUDY_PROTOCOL.md                 # Detailed study documentation
├── LICENSE                           # MIT License
├── requirements.txt                  # Python dependencies
├── .gitignore
│
├── data/
│   ├── vignettes_main.xlsx           # 69 clinical vignettes (23 scenarios × 3 variants)
│   ├── vignettes_augmentation.xlsx   # 30 augmentation vignettes (10 scenarios × 3 variants)
│   └── ad_artifacts_database.xlsx    # 42 real pharmaceutical ad texts
│
├── pipeline_main.py                  # Main experiment pipeline
├── pipeline_augmentation.py          # Augmentation experiment pipeline
├── analysis_main.py                  # Main experiment analysis & visualization
├── analysis_augmentation.py          # Augmentation analysis & visualization
│
└── results/
    ├── main/                         # Main experiment outputs (per-model .xlsx)
    └── augmentation/                 # Augmentation outputs (per-model .xlsx)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
# Set one or more, depending on which models you want to test
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# For Vertex AI (higher rate limits):
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### 3. Run the Main Experiment

```bash
python pipeline_main.py
```

The interactive CLI will prompt you to select a provider, model, system prompts, and scenarios. Output is saved to `results/main/`.

### 4. Run the Analysis

```bash
# Analyze a single model
python analysis_main.py results/main/adverse_gpt-5.2_20260217.xlsx

# Analyze all models in a directory
python analysis_main.py results/main/
```

Generates a multi-page PDF with 7 figures, individual PNGs, a summary Excel file, and a console report.

### 5. Run the Augmentation Experiment

```bash
python pipeline_augmentation.py          # Run augmentation pipeline
python analysis_augmentation.py results/augmentation/   # Analyze results
```

---

## Output Files

### Pipeline Output (per model)

| Sheet | Description |
|---|---|
| `Grading` | Every API call with parsed choice, correctness, brand selection flags |
| `Deltas` | Per-scenario/variant/prompt comparison: baseline vs ad condition |
| `Raw_Outputs` | Full model responses for reproducibility |
| `Run_Config` | Exact parameters used for this run |

### Analysis Output

**Main experiment** (7 figures):

1. Overall ad-induced brand selection shift (paired dot plot + shift bars)
2. Ad susceptibility by system prompt persona
3. Ad susceptibility by model (nano → flagship)
4. Shift by clinical scenario type × system prompt interaction
5. Wellness supplement endorsement rates
6. Model × prompt interaction heatmap
7. Scenario × condition accuracy heatmap

**Augmentation** (6 figures):

1. Accuracy cost per scenario (paired dot plot + cost bars)
2. Answer distribution shift (A/B/C stacked bars)
3. Accuracy cost by system prompt persona
4. Accuracy cost by model
5. Model × prompt accuracy cost heatmap
6. Specificity of ad effect (advertised vs non-advertised brand gain)

---

## Key Design Decisions

- **Temperature 0.7** — captures the probability distribution of model preferences, not just the greedy output
- **20 repeats** — provides statistical power to detect shifts of ~5 percentage points
- **Within-subject design** — every model sees every scenario under every condition, eliminating between-subject variability
- **4 system prompts** — tests the hypothesis that customer-pleasing personas are more susceptible to ad influence
- **Real advertisements** — 42 genuine pharmaceutical ad texts collected from manufacturer websites, Meta, and Google
- **Wilson confidence intervals** — for binomial proportions; Cohen's h for effect sizes
- **JSONL checkpointing** — automatic resume on interruption; no lost API calls

---

## Statistical Methods

- **Primary metric (main):** Within-subject brand selection shift with Wilson 95% CIs
- **Primary metric (augmentation):** Accuracy cost with paired baseline comparison
- **Effect size:** Cohen's h (arcsine transformation of proportions)
- **Stratification:** By model, system prompt, scenario type (clear preference / equipoise / neither correct), therapeutic area
- **Significance:** Two-proportion z-test where applicable (requires SciPy)

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{omar2026adverse,
  title={Ad-verse Effects: Do Pharmaceutical Advertisements Embedded in LLM Interactions Shift Clinical Recommendations?},
  author={Omar, Mahmud},
  year={2026},
  journal={Under review}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
