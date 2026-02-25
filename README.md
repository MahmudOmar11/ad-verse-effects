<p align="center">
  <img src="assets/banner.svg" alt="Ad-verse Effects" width="100%"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"/></a>
  <img src="https://img.shields.io/badge/models-12_LLMs-e94560?style=flat-square" alt="12 Models"/>
  <img src="https://img.shields.io/badge/API_calls-256K-302b63?style=flat-square" alt="256K API Calls"/>
  <img src="https://img.shields.io/badge/providers-3-14B8A6?style=flat-square" alt="3 Providers"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/></a>
</p>

---

## Overview

Large language models are increasingly deployed as clinical decision-support tools and patient-facing health chatbots. In January 2026, OpenAI announced plans to introduce advertising into ChatGPT; Microsoft already embeds ads in Copilot. Whether pharmaceutical advertisements can alter the clinical outputs of these systems has not been tested.

This repository contains the full experimental code, clinical vignettes, and analysis pipeline for a controlled study measuring the effect of pharmaceutical advertising on LLM drug recommendations. We embedded standardized brand advertisements into the system prompts of 12 LLMs from OpenAI, Anthropic, and Google and measured their effect across **256,320 API calls** spanning three experiments.

---

## Key Findings

Advertising shifted drug recommendations substantially, systematically, and silently. Models selected the advertised drug 47.6% of the time versus ~33% at baseline — a +12.7 percentage point shift overall. Google models shifted by up to +29.8 pp, OpenAI by +10.9 pp, and Anthropic by +2.0 pp. Multiple model–scenario combinations showed complete 0→100% shifts (e.g., dapagliflozin in heart failure, eszopiclone in insomnia, semaglutide in type 2 diabetes). This bias operated between clinically equivalent alternatives without reducing response accuracy — the user receives a biased recommendation that appears objective.

---

## Three Experiments

### Experiment 1 — Preference Shift (S01–S13)

All drug options are clinically appropriate. There is no wrong answer. Does an ad shift which drug the model recommends?

- 13 Rx scenarios, 4 branded options each
- 3 ad conditions: baseline, ad for drug A, ad for drug B
- Metric: brand selection shift (pp)

### Experiment 2 — Wellness Endorsement (S14–S23)

Each scenario presents a branded supplement versus evidence-based advice. Does an ad push the model toward a product over proven interventions?

- 10 wellness scenarios, supplement (A) vs evidence-based (B)
- 2 ad conditions: baseline, ad for supplement
- Metric: supplement endorsement rate shift

### Experiment 3 — Accuracy Cost (A01–A10)

The advertised drugs are now clinically suboptimal. The correct answer is always the non-advertised option. Does the model sacrifice accuracy?

- 10 augmentation scenarios, 3 options (A, B advertised; C correct)
- 3 ad conditions: baseline, ad for A, ad for B
- Metric: accuracy cost = P(correct | baseline) − P(correct | ad)

---

## Factorial Design

Every model is its own control. Each model–scenario–condition combination is repeated 20 times at temperature 0.7 to capture the stochastic distribution of preferences.

| Factor | Levels |
|:---|:---|
| Models | 12 (5 OpenAI, 4 Anthropic, 3 Google) |
| System prompt | 4 personas: physician, helpful AI, customer service, no persona |
| Ad condition | 2–3 per scenario (baseline, ad_a, ad_b) |
| Vignette variant | 3 independently worded versions per scenario |
| Repetitions | 20 per condition (temperature = 0.7) |

Total: 256,320 API calls (169,920 main + 86,400 augmentation).

---

## Models Tested

| Provider | Models | N |
|:---|:---|:---:|
| **OpenAI** | GPT-4.1 Mini · GPT-4.1 · GPT-5 Mini · GPT-5.2 · o4-mini | 5 |
| **Anthropic** | Claude Haiku 4.5 · Claude Sonnet 4.5 · Claude Sonnet 4.6 · Claude Opus 4.6 | 4 |
| **Google** | Gemini 2.5 Flash-Lite · Gemini 2.5 Flash · Gemini 3 Flash Preview | 3 |

---

## Repository Structure

```
ad-verse-effects/
├── README.md                          # This file
├── STUDY_PROTOCOL.md                  # Full study protocol
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── vignettes_main.xlsx            # 69 clinical vignettes (23 scenarios × 3 variants)
│   ├── vignettes_augmentation.xlsx    # 30 vignettes (10 scenarios × 3 variants)
│   └── ad_artifacts_database.xlsx     # 42 real pharmaceutical ad texts
│
├── pipeline_main.py                   # Experiment 1 + 2 pipeline
├── pipeline_augmentation.py           # Experiment 3 pipeline
├── analysis_main.py                   # Analysis: preference shift + wellness
├── analysis_augmentation.py           # Analysis: accuracy cost
├── generate_figures.py                # Publication-ready figure generation
│
├── results/
│   ├── main/                          # Per-model output (Experiments 1–2)
│   └── augmentation/                  # Per-model output (Experiment 3)
│
└── assets/
    └── banner.svg
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/MahmudOmar11/ad-verse-effects.git
cd ad-verse-effects
pip install -r requirements.txt

# Set API keys (one or more providers)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Run experiments
python pipeline_main.py                  # Experiments 1 + 2 (interactive model selection)
python pipeline_augmentation.py          # Experiment 3

# Analyze results
python analysis_main.py results/main/
python analysis_augmentation.py results/augmentation/

# Generate publication figures
python generate_figures.py results/main/ results/augmentation/
```

For Google models via Vertex AI (higher rate limits):
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

---

## Output

Each pipeline run produces a per-model Excel workbook with four sheets:

| Sheet | Contents |
|:---|:---|
| **Grading** | Every API call with parsed choice, correctness, and brand selection flags |
| **Deltas** | Paired baseline vs. ad comparisons per scenario, variant, and prompt |
| **Raw_Outputs** | Full model responses for reproducibility |
| **Run_Config** | Exact parameters, timestamps, and model identifiers |

The analysis scripts produce summary tables (CSV + Excel), multi-page PDF figures, and console reports with statistical tests.

---

## Design Rationale

**Temperature 0.7** captures the probability distribution of preferences, not just greedy output. **20 repeats × 3 variants** = 60 observations per condition, powering detection of ~5 pp shifts. **Within-subject** design means every model is its own control. **Real ads** — 42 genuine pharmaceutical advertisement texts sourced from manufacturer websites, Meta Ad Library, and Google Ads. **Clearly labeled** — all ads are wrapped in `[paid advertisement]` tags to test whether even disclosed advertising shifts behavior.

---

## Citation

```bibtex
@article{omar2026adverse,
  title   = {Ad-verse Effects: Pharmaceutical Advertisements Shift Clinical
             Recommendations of Large Language Models},
  author  = {Omar, Mahmud},
  year    = {2026},
  journal = {Under review}
}
```

---

<p align="center">
  <sub>MIT License · Mahmud Omar, MD · BRIDGE GenAI Lab, BIDMC · Mount Sinai Medical Center</sub>
</p>
