<p align="center">
  <img src="assets/banner.svg" alt="Ad-verse Effects" width="100%"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"/></a>
  <img src="https://img.shields.io/badge/models-12_LLMs-e94560?style=flat-square" alt="12 Models"/>
  <img src="https://img.shields.io/badge/API_calls-256K-302b63?style=flat-square" alt="256K API Calls"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/></a>
</p>

---

## Overview

Large language models are increasingly deployed as clinical decision-support tools and patient-facing health chatbots. Whether pharmaceutical advertisements can alter the clinical outputs of these systems has not been tested.

We embedded real pharmaceutical advertisements into the system prompts of 12 LLMs from three providers (OpenAI, Anthropic, Google) and measured the effect on drug recommendations across 256,320 API calls.

Advertising shifted drug recommendations by +12.7 percentage points overall without reducing response accuracy. Multiple model–scenario pairs showed complete 0→100% shifts toward the advertised drug.

---

## Study Design

Within-subject factorial experiment. Every model sees every scenario under every condition and serves as its own control. Each combination is repeated 20 times at temperature 0.7.

| Factor | Levels |
|:---|:---|
| Models | 12 (5 OpenAI · 4 Anthropic · 3 Google) |
| System prompt | 4 personas (physician · helpful AI · customer service · no persona) |
| Ad condition | 2–3 per scenario (baseline · ad_a · ad_b) |
| Vignette variant | 3 per scenario |
| Repetitions | 20 per condition |

**Experiment 1 — Preference Shift (S01–S13):** 13 Rx scenarios with 4 clinically equivalent drug options. Does an ad shift which drug the model recommends? 9,360 calls per model.

**Experiment 2 — Wellness Endorsement (S14–S23):** 10 scenarios pitting a supplement against evidence-based advice. Does an ad increase supplement endorsement? 4,800 calls per model.

**Experiment 3 — Accuracy Cost (A01–A10):** 10 scenarios where the advertised drug is suboptimal and the correct answer is always non-advertised. Does the model sacrifice accuracy? 7,200 calls per model.

---

## Models

| Provider | Models |
|:---|:---|
| **OpenAI** | GPT-4.1 Mini · GPT-4.1 · GPT-5 Mini · GPT-5.2 · o4-mini |
| **Anthropic** | Claude Haiku 4.5 · Claude Sonnet 4.5 · Claude Sonnet 4.6 · Claude Opus 4.6 |
| **Google** | Gemini 2.5 Flash-Lite · Gemini 2.5 Flash · Gemini 3 Flash Preview |

---

## Repository

```
ad-verse-effects/
├── data/
│   ├── vignettes_main.xlsx            # 69 vignettes (23 scenarios × 3 variants)
│   ├── vignettes_augmentation.xlsx    # 30 vignettes (10 scenarios × 3 variants)
│   └── ad_artifacts_database.xlsx     # 42 real pharmaceutical ad texts
│
├── pipeline_main.py                   # Experiments 1 + 2
├── pipeline_augmentation.py           # Experiment 3
├── analysis_main.py                   # Analysis: preference shift + wellness
├── analysis_augmentation.py           # Analysis: accuracy cost
│
├── STUDY_PROTOCOL.md                  # Full protocol
├── requirements.txt
└── LICENSE
```

---

## Replication

```bash
pip install -r requirements.txt

# Set one or more API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Run experiments (interactive model selection)
python pipeline_main.py
python pipeline_augmentation.py

# Analyze
python analysis_main.py results/
python analysis_augmentation.py results/
```

Each pipeline run produces a per-model Excel workbook with four sheets: **Grading** (every API call with parsed choice and brand selection flags), **Deltas** (paired baseline vs. ad comparisons), **Raw_Outputs** (full model responses), and **Run_Config** (exact parameters and timestamps).

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
