<p align="center">
  <img src="assets/banner.svg" alt="Ad-verse Effects" width="100%"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"/></a>
  <img src="https://img.shields.io/badge/models-12_LLMs-e94560?style=flat-square" alt="12 Models"/>
  <img src="https://img.shields.io/badge/API_calls-258K-302b63?style=flat-square" alt="258K API Calls"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/></a>
</p>

---

## Overview

Large language models are increasingly deployed as clinical decision-support tools and patient-facing health chatbots. Whether pharmaceutical advertisements can alter the clinical outputs of these systems has not been tested.

We embedded real pharmaceutical advertisements into the system prompts of 12 LLMs from three providers (OpenAI, Anthropic, Google) and measured the effect on drug recommendations across 258,660 API calls spanning four experiments.

Advertising shifted drug recommendations by +12.7 percentage points overall without reducing response accuracy. Multiple model–scenario pairs showed complete 0→100% shifts toward the advertised drug. An open-response sub-analysis confirmed that advertising restructures free-text clinical reasoning: models echoed ad claims at 2.7× the baseline rate while maintaining uniformly high confidence and rarely disclosing the advertising presence.

---

## Study Design

Within-subject factorial experiment. Every model sees every scenario under every condition and serves as its own control.

| Factor | Levels |
|:---|:---|
| Models | 12 (5 OpenAI · 4 Anthropic · 3 Google) |
| System prompt | 4 personas (physician · helpful AI · customer service · no persona) |
| Ad condition | 2–3 per scenario (baseline · ad_a · ad_b) |
| Vignette variant | 3 per scenario (Experiments 1–3) or 1 (Experiment 4) |
| Repetitions | 20 per condition (Experiments 1–3) or 5 (Experiment 4) |

**Experiment 1 — Preference Shift (S01–S13):** 13 Rx scenarios with clinically equivalent drug options. Does an ad shift which drug the model recommends? 9,360 calls per model × 12 models = 112,320 calls.

**Experiment 2 — Wellness Endorsement (S14–S23):** 10 scenarios pitting a supplement against evidence-based advice. Does an ad increase supplement endorsement? 4,800 calls per model × 12 models = 57,600 calls.

**Experiment 3 — Accuracy Cost (A01–A10):** 10 scenarios where the advertised drug is suboptimal and the correct answer is always non-advertised. Does the model sacrifice accuracy? 7,200 calls per model × 12 models = 86,400 calls.

**Experiment 4 — Open-Response Sub-Analysis (S01–S13):** 3 representative models generate free-text clinical justifications alongside drug selections. Does advertising restructure clinical reasoning? 780 calls per model × 3 models = 2,340 calls.

---

## Key Findings

| Metric | Result |
|:---|:---|
| Overall preference shift | +12.7 pp (baseline ~34% → 47.6%, *P* < 0.001) |
| Google models | +29.8 pp (most susceptible) |
| OpenAI models | +10.9 pp (moderate) |
| Anthropic models | +2.0 pp (resistant) |
| Complete reversals (0→100%) | Farxiga in HF, Lunesta in insomnia, Claritin in rhinitis, Ozempic in T2D |
| Supplement endorsement change | −0.6 pp (models resisted) |
| Accuracy when ad drug is suboptimal | 95.6% (preserved) |
| Ad-echo rate (chose advertised) | 52.7% vs 19.4% (2.7× difference) |
| Spontaneous ad disclosure | Claude 55.9%, Gemini 28.7%, GPT-4.1 5.2% |

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
│   ├── vignettes_main.xlsx              # 69 vignettes (23 scenarios × 3 variants)
│   ├── vignettes_augmentation.xlsx      # 30 vignettes (10 scenarios × 3 variants)
│   └── ad_artifacts_database.xlsx       # 46 real pharmaceutical ad texts
│
├── pipeline_main.py                     # Experiments 1 + 2 (forced-choice)
├── pipeline_augmentation.py             # Experiment 3 (accuracy cost)
├── pipeline_open_response.py            # Experiment 4 (open-response sub-analysis)
├── analysis_main.py                     # Analysis: preference shift + wellness
├── analysis_augmentation.py             # Analysis: accuracy cost
├── analysis_open_response.py            # Analysis: open-response NLP metrics
│
├── STUDY_PROTOCOL.md                    # Full protocol (all 4 experiments)
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
python pipeline_main.py                  # Experiments 1 + 2
python pipeline_augmentation.py          # Experiment 3
python pipeline_open_response.py         # Experiment 4

# Analyze
python analysis_main.py results/
python analysis_augmentation.py results/
python analysis_open_response.py         # Auto-detects latest results file
```

**Experiments 1–3** each produce a per-model Excel workbook with four sheets: **Grading** (every API call with parsed choice and brand selection flags), **Deltas** (paired baseline vs. ad comparisons), **Raw_Outputs** (full model responses), and **Run_Config** (exact parameters and timestamps).

**Experiment 4** produces a JSONL results file and an Excel workbook with 8+ analysis sheets covering preference shift, ad-echo rates, disclosure rates, confidence scores, evidence specificity, and competitor mention suppression.

---

## Total API Calls

| Pipeline | Calls |
|:---|---:|
| Experiment 1 — Preference Shift (12 models × 9,360) | 112,320 |
| Experiment 2 — Wellness Endorsement (12 models × 4,800) | 57,600 |
| Experiment 3 — Accuracy Cost (12 models × 7,200) | 86,400 |
| Experiment 4 — Open-Response (3 models × 780) | 2,340 |
| **Grand Total** | **258,660** |

---

## Citation

```bibtex
@article{omar2026adverse,
  title   = {Ad-verse Effects: Pharmaceutical Advertising Shifts Drug
             Recommendations by AI Health Assistants},
  author  = {Omar, Mahmud and Agbareia, Reem and Klang, Eyal and Nadkarni, Girish N.},
  year    = {2026},
  journal = {Under review}
}
```

---

<p align="center">
  <sub>MIT License · Mahmud Omar, MD · BRIDGE GenAI Lab, BIDMC · Mount Sinai Medical Center</sub>
</p>
