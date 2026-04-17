# Part 2 Report Outline (NeurIPS 4-page, Internal Data Only)

## Compliance Statement
- This report uses only competition-provided files from the repository.
- No external datasets, no external labels, no third-party enrichment.
- All charts are reproducible from notebook `Data/baseline.ipynb`.

## Suggested 4-page structure

### Page 1: Business Context + Data Quality
- Problem framing:
  - Forecast-ready business diagnosis for revenue, operations, and profitability.
  - Scope: 2012-07-04 to 2022-12-31 (train period in provided data).
- Data layers used:
  - Master, Transaction, Analytical, Operational.
- Data quality audit:
  - Missingness, duplicate check, time coverage consistency.
- Figure placement:
  - `Results/figures/fig_01_data_quality_missing_pct.png`

### Page 2: Demand Pattern and Seasonality (Revenue Drivers)
- Monthly demand evolution and structural break observation.
- Seasonality findings:
  - Highest average month: 5.
  - Highest average day of week: Wednesday.
- Business interpretation:
  - Strong seasonal cycles imply inventory and campaign timing must be calendar-aligned.
- Figure placement:
  - `Results/figures/fig_02_monthly_trends.png`
  - `Results/figures/fig_03_seasonality.png`

### Page 3: Commercial Efficiency (Promo + Channel)
- Promo quality:
  - Promo orders have lower AOV than non-promo in current data snapshot.
  - Segment margin differences indicate uneven profitability.
- Channel quality matrix (efficiency vs risk):
  - Best conversion efficiency proxy: organic_search (orders per 1,000 sessions).
  - Highest return-rate channel: email_campaign (~3.46%).
- Action point:
  - Shift spend toward high-efficiency channels, tighten targeting/creative on high-return channels.
- Figure placement:
  - `Results/figures/fig_04_promo_and_margin.png`
  - `Results/figures/fig_05_channel_matrix.png`

### Page 4: Operational Risk (Inventory + Returns) and Recommendations
- Inventory health:
  - Latest snapshot overstock risk highest in GenZ (~84.09%).
- Return and leakage:
  - Highest return-rate category: GenZ (~3.52%).
  - Largest refund-to-gross-profit leakage: Casual (~0.4157 ratio).
- Priority recommendations (90-day plan):
  - Rebalance safety stock and reorder policy by category.
  - Apply return-aware marketing and tighter quality control for high-risk categories.
  - Add margin guardrails for promo decisions.
- Figure placement:
  - `Results/figures/fig_06_inventory_risk.png`
  - `Results/figures/fig_07_return_and_profit_leakage.png`

## Ready-to-use narrative bullets
- Demand is seasonal and concentrated around late spring/early summer; campaign and procurement calendars should be synchronized with this peak.
- Channel performance is heterogeneous: some channels convert better but bring higher return risk, requiring channel-level quality controls.
- Inventory stress is persistent (overstock and stockout coexist), signaling misalignment between replenishment and sell-through behavior.
- Return behavior creates measurable profit leakage; category-specific interventions are needed rather than one-size-fits-all policies.

## Appendix suggestion (optional, not counted in 4 pages)
- Detailed feature definitions and formulas.
- Additional robustness checks by year and category.
- Reproducibility note: notebook execution order and environment.
