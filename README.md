# Gold Price Analysis

A collection of Python scripts for analyzing gold price dynamics, macroeconomic drivers, and precious metal market structure.

## Charts & Scripts

| Script | Description | Chart |
|--------|-------------|-------|
| `CFTC黄金持仓分析.py` | CFTC Gold Positioning Analysis — 4-panel COT report: net speculative positions, open interest, non-commercial long/short, commercial net hedging (2008–present) | `CFTC黄金持仓分析_chart.png` |
| `GOFO计算.py` | GOFO (Gold Forward Offered Rate) Calculation — 90-day constant-maturity interpolation using COMEX monthly futures contracts (GCF–GCZ) | `GOFO计算_chart.png` |
| `黄金租借利率.py` | Gold Lease Rate — Risk-free rate (LIBOR/SOFR) minus GOFO_90 | `黄金租借利率_chart.png` |
| `黄金远期曲线.py` | Gold Forward Curve — Historical term structure on 6 selected dates; % premium and implied annualized rate | `黄金远期曲线_chart.png` |
| `贵金属价格与波动率.py` | Precious Metals Price & Volatility — 2×2 grid: XAU/XAG prices with converging triangle patterns + GVZ/VXSLV volatility indices with low-volatility shading | `贵金属价格与波动率_chart.png` |
| `金银比与利差.py` | Gold/Silver Ratio vs. Yield Spread — Rolling Spearman correlation (36M & 12M windows) | `金银比与利差_chart.png` |
| `货币与利率.py` | Currency & Interest Rates — JPY/EUR rates, USD Index, GVZ, VXN/VXD, and 10Y–3M yield spread | `货币与利率_chart.png` |
| `黄金与原油.py` | Gold vs. Crude Oil — Oil/Gold ratio and real interest rates | `黄金与原油_chart.png` |
| `综合分析因素.py` | Comprehensive Factor Analysis — Multi-panel macro drivers (1962–present, split into two eras) | `综合分析因素_chart.png` |
| `美债与美联储资产负债表.py` | US Treasury Debt & Fed Balance Sheet | `美债与美联储资产负债表.png` |
| `联邦利息负担.py` | Federal Interest Burden — Interest payments as % of revenue | `联邦利息负担.png` |
| `利率病时期.py` | Low-Rate Era Analysis — Leverage ratio and Gini coefficient | `利率病时期_杠杆率与基尼系数.png` |
| `货币与利率.py` | Currency & Rates — Annotated with key policy events (Abenomics, QQE, Fed cycles) | `货币与利率_chart.png` |
| `日中国际收支比较.py` | Japan–China Balance of Payments Comparison | `日中国际收支比较.png` |
| `中美贸易与制造业.py` | US–China Trade & Manufacturing | `中美贸易与制造业.png` |
| `gold-realrate.py` | Gold vs. Real Interest Rates | `gold_realrate_chart.png` |
| `gold-realpurchasepower.py` | Gold Real Purchasing Power | `gold_realpurchasepower_chart.png` |
| `gold-quarterly-change-rate.py` | Gold Quarterly Change Rate | `gold_quarterly_changerate_chart.png` |
| `real interest-real gold price change.py` | Real Interest Rate vs. Real Gold Price Change | `real_gold_rate_change_chart.png` |

## Key Methodologies

- **GOFO 90-day Constant Maturity**: Near/Far contract pair bracketing 90 days interpolated linearly; contracts within 45 days of expiry excluded (GARBAGE_DAYS=45)
- **Gold Lease Rate**: LIBOR (pre-2018-07-02) / SOFR (post) minus GOFO_90
- **CFTC COT**: Legacy format; mini-contract rows filtered out (OI < 100,000)
- **Converging Triangles**: Auto-detected via `scipy.signal.find_peaks` + linear regression on peaks/troughs; convergence = upper slope < lower slope
- **Spearman Rolling Correlation**: 36-month (deep purple) and 12-month (cyan) windows on monthly resampled data

## Data Sources

- [CFTC Legacy COT Reports](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)
- [CBOE GVZ / VXSLV](https://www.cboe.com/tradable_products/volatility/)
- [FRED](https://fred.stlouisfed.org/) — SOFR, DGS5, T10Y3M, Federal Funds Rate
- [Barchart](https://www.barchart.com/) — COMEX Gold Futures (GCF–GCZ, 2000–2027)
- [Investing.com](https://www.investing.com/) — XAU/USD, XAG/USD weekly prices

## Requirements

```bash
pip install pandas numpy matplotlib scipy
```
