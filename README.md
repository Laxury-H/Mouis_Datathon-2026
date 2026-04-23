# Mouis Datathon 2026

![Competition](https://img.shields.io/badge/Datathon-2026-blue)
![Track](https://img.shields.io/badge/Track-Ecommerce%20Analytics-0A7EA4)
![Report](https://img.shields.io/badge/Report-NeurIPS%202025%20Format-4C1)
![Status](https://img.shields.io/badge/Status-Active-success)

Production-quality competition repository for Datathon 2026: The Gridbreakers, covering business analytics, strategic storytelling, and reproducible reporting artifacts.

## Executive Summary

This project addresses a simulated Vietnamese fashion e-commerce business problem with three competition components:
- Part 1 (MCQ): Objective business/data diagnostic questions.
- Part 2 (EDA and strategy): NeurIPS-formatted, decision-oriented analysis.
- Part 3 (forecasting): Daily revenue prediction on Kaggle.

Current repository maturity:
- Part 1: Completed and documented.
- Part 2: Completed, compressed, and compiled to PDF preview.
- Part 3: Ready for modeling pipeline expansion.

## Competition Context

Scoring breakdown:
- Part 1: 20 points
- Part 2: 60 points
- Part 3: 20 points

Data horizon:
- Training period: 2012-07-04 to 2022-12-31
- Forecast period: 2023-01-01 to 2024-07-01

Hard constraints:
- No external data.
- Full reproducibility.
- Explainability in business language.

## Repository Layout

```text
Mouis_Datathon-2026/
  Data/
    customers.csv
    geography.csv
    inventory.csv
    order_items.csv
    orders.csv
    payments.csv
    products.csv
    promotions.csv
    returns.csv
    reviews.csv
    sales.csv
    sample_submission.csv
    shipments.csv
    web_traffic.csv

  Documents/
    Overview.md
    Rules.md
    data.md

  Results/
    history/
      README.md
      submissions/
        v17/
          README.md
        v18/
          README.md
    part1_results.md
    part2_neurips_stories.tex
    preview_part2.tex
    preview_part2.pdf
    neurips_2025.sty
    figures/
    submissions/
      v18/

  README.md
```

## Key Outputs

Part 1 deliverable:
- Results/part1_results.md

Part 2 deliverables:
- Results/part2_neurips_stories.tex
- Results/preview_part2.tex
- Results/preview_part2.pdf

Part 2 narrative quality standard:
- Three end-to-end business stories.
- Explicit Descriptive, Diagnostic, Predictive, and Prescriptive logic in each story.
- Quantified business impacts and action paths.
- Academic figure captions aligned with NeurIPS writing style.

## Quick Start

### 1) Prerequisites

- Windows PowerShell
- MiKTeX (or any LaTeX distribution with pdflatex)

### 2) Build the Part 2 Preview PDF

From the Results folder:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error preview_part2.tex
pdflatex -interaction=nonstopmode -halt-on-error preview_part2.tex
```

Expected artifact:
- preview_part2.pdf

## Reproducibility and Data Governance

- All Part 2 metrics are computed strictly from provided competition data in Data/.
- No external enrichment sources are used.
- Figure placeholders in the report reference:
  - Results/figures/story1_channel_matrix.png
  - Results/figures/story2_discount_margin.png
  - Results/figures/story3_inventory_return_risk.png
- Local NeurIPS style support is included at Results/neurips_2025.sty.

## Submission Readiness Checklist

- Part 1 MCQ answers finalized and reviewed.
- Part 2 report fits page budget and passes narrative coherence check.
- Part 3 submission.csv preserves sample_submission row ordering.
- Repository is accessible to judges.
- Final commit is tagged for traceability.

## Engineering Workflow Recommendation

- Keep analysis source code in dedicated notebooks/src folders.
- Keep all submission artifacts centralized in Results/.
- Archive old submission families in Results/history/ so the active Data/ folder stays small.
- Freeze a submission branch before deadline.
- Tag the exact commit used for PDF and Kaggle upload.

## Roadmap (Post-Part 2)

- Implement Part 3 forecasting baseline and advanced models.
- Add experiment tracking for MAE, RMSE, and R2.
- Add explainability block for final technical report section.

## Compliance Notice

This repository is for Datathon 2026 participation. Usage and redistribution must comply with official competition rules in Documents/Rules.md.