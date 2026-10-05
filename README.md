# ReLoop: Move the Decision, Not the Asset

**AI-enabled reverse logistics for corporate IT assets and e-waste in India**

Working with AI (WAI) project · Logistics & Warehousing Management, Term V · Executive MBA 2025-27, IIM Ranchi
**Author:** Himanshu Rai (XW013-25) · **Faculty:** Prof. Krishna Kumar Dadsena

---

## The question

How should a corporate IT-asset-disposition (ITAD) network serving BFSI and GCC clients in 12 Indian cities be
redesigned, and which AI capabilities does it need, to recover more value with less freight, carbon and data risk?

## Key results

| Finding | Number | Source |
|---|---|---|
| Licensed e-waste recycling capacity (411 recyclers, Aug 2026) | 42.3 lakh t / yr | CPCB registry |
| Share of that capacity used in FY 2024-25 | 23.4% | CPCB registry + PIB |
| Share of formal processing in UP, Uttarakhand, Haryana | 65.9% | PIB |
| Logistics cost, 3 regional hubs vs one national vendor | −40.7% (₹7.41 Cr → ₹4.39 Cr) | MILP, OR-Tools |
| Transport CO₂e / data-bearing tonne-km | −80.1% / −80.8% | MILP, OR-Tools |
| Value recovered per asset, AI grading vs age rule | +₹826 (+15%) | Random Forest |
| Five-year NPV, base case (₹5.1 Cr capex) | ₹55.7 Cr | Cost-benefit model |
| Top risks (practitioner fuzzy FMEA) | Informal leakage, refresh-cycle surges, transit data breach | Survey, n = 3 used |

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                  # ~1 minute; regenerates every table, figure and results.json
open dashboard/ReLoop_Control_Tower.html
```

> **Reproducibility note.** Library versions are pinned in `requirements.txt` and synthetic inputs are frozen in
> `data/synthetic/`. Random Forest results can still differ by about 1–2% across CPU architectures; the submitted
> numbers come from an Apple Silicon (M-series) run.

## Repository structure

```
├── run_all.py                     # runs every module in order
├── requirements.txt               # pinned versions
├── data/
│   ├── raw/                       # REAL public data + practitioner survey
│   └── synthetic/                 # SYNTHETIC inputs, frozen (disclosed)
├── src/
│   ├── config.py                  # paths, palette, distance helpers, results logger
│   ├── a01_real_diagnostics.py    # registry cleaning, mismatch index, capacity atlas
│   ├── a01b_state_typology.py     # K-means typology of 16 regional e-waste systems
│   ├── a02_forecast.py            # logistic S-curve + EPR policy scenarios to FY30
│   ├── a03_disposition_ml.py      # Random Forest grading engine vs rule-based policy
│   ├── a04_network_design.py      # K-means regions + MILP facility location + sensitivity
│   ├── a05_milkrun_vrp.py         # capacitated vehicle routing (Pune milk-runs)
│   ├── a06_fuzzy_fmea.py          # fuzzy FMEA on practitioner ratings + robustness
│   ├── a07_cost_benefit.py        # NPV, payback, benefit scenarios
│   ├── a08_flowchart_gantt.py     # methodology flowchart and roadmap
│   └── a09_build_dashboard.py     # offline HTML control-tower dashboard
├── dashboard/                     # ReLoop_Control_Tower.html (runs offline)
└── outputs/
    ├── figures/  tables/          # every chart and model table
    └── results.json               # single source of truth for numbers in the report
```

## Data disclosure

- **Real public data:** CPCB EPR-portal recycler registry (413 rows, 2 test records removed → 411); Lok Sabha
  USQ 983 (2020) and USQ 2458 (2021); PIB releases 1943201, 1941054 and 2147876 (national and state processing
  FY18-FY25); TRAI tele/internet density, Sep 2024 (via Jharkhand Economic Survey 2025-26, Table 9.8); Census 2011.
- **Proxy:** state e-waste generation = TRAI internet density × Census population. GSDP share is used only as a
  robustness check (rank correlation 0.99).
- **Real practitioner input:** anonymous fuzzy-FMEA survey, Oct 2026 (`data/raw/fmea_practitioner_survey_oct2026.csv`);
  4 responses, 1 excluded for straight-lining. An earlier LLM-simulated panel is kept in code only as a benchmark.
- **Synthetic (only where real data is confidential):** 12,000 IT-asset return records, 398 client pickup sites and
  20 Pune pickup points, generated from documented logic and frozen.
- **Assumptions:** freight rates, hub costs, asset values and emission factors are stated in the report (Annex C)
  and stress-tested in `outputs/tables/t04_sensitivity.csv`.

## AI use

Claude (Anthropic) was used for scoping, literature mapping, code drafting and red-team critique of results. Every
output was reviewed and run by the author; the full Prompt Logbook is submitted with the report.
