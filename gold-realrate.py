import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 1. Load data ───────────────────────────────────────────────────────────────
gold = pd.read_csv("monthly-gold-price.csv")
gold["Date"] = pd.to_datetime(gold["Date"])

real_rate = pd.read_csv("real_interest_rate.csv")
real_rate["Date"] = pd.to_datetime(real_rate["Date"])

# ── 2. Resample gold price to quarterly ───────────────────────────────────────
gold_q = (gold.set_index("Date")["Price"]
              .resample("QS").last()
              .reset_index()
              .rename(columns={"Price": "Gold_Price"}))

# ── 3. Merge ───────────────────────────────────────────────────────────────────
df = (gold_q.merge(real_rate[["Date", "Real_DGS10", "Real_DGS5", "Real_FF"]], on="Date")
            .sort_values("Date")
            .reset_index(drop=True))

# ── 4. Plot ────────────────────────────────────────────────────────────────────
COLOR_GOLD  = "#D4A017"
COLOR_DGS10 = "#1565C0"
COLOR_DGS5  = "#66BB6A"
COLOR_FF    = "#D32F2F"

num_years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
fig, ax1 = plt.subplots(figsize=(max(20, num_years * 0.55), 9))

# — Left axis: Gold Price —
ax1.plot(df["Date"], df["Gold_Price"],
         color=COLOR_GOLD, linewidth=1.2, label="Gold Price (USD)")
ax1.fill_between(df["Date"], df["Gold_Price"], 0,
                 alpha=0.15, color=COLOR_GOLD)
ax1.set_ylabel("Gold Price (USD)", color=COLOR_GOLD, fontsize=15)
ax1.tick_params(axis="y", labelcolor=COLOR_GOLD, labelsize=13)

# — Right axis: 3 real interest rates —
ax2 = ax1.twinx()

# 图层叠加填充：top→0, mid→0, bot→0，后层覆盖前层自然形成带状，无缝隙
fill_df = df.dropna(subset=["Real_DGS10", "Real_DGS5", "Real_FF"])
dates = fill_df["Date"]
vals  = np.sort(np.stack([fill_df["Real_DGS10"].values,
                           fill_df["Real_DGS5"].values,
                           fill_df["Real_FF"].values], axis=1), axis=1)
bot, mid, top = vals[:, 0], vals[:, 1], vals[:, 2]

ax2.fill_between(dates, top, 0, color=COLOR_DGS10, alpha=0.2, linewidth=0)
ax2.fill_between(dates, mid, 0, color=COLOR_DGS5,  alpha=0.2, linewidth=0)
ax2.fill_between(dates, bot, 0, color=COLOR_FF,    alpha=0.2, linewidth=0)

for col, color, label in [
    ("Real_DGS10", COLOR_DGS10, "Real Rate — DGS10 (10Y Treasury − CPI YoY, %)"),
    ("Real_DGS5",  COLOR_DGS5,  "Real Rate — DGS5  (5Y Treasury − CPI YoY, %)"),
    ("Real_FF",    COLOR_FF,    "Real Rate — Fed Funds (FF − CPI YoY, %)"),
]:
    s = df.dropna(subset=[col])
    ax2.plot(s["Date"], s[col], color=color, linewidth=1.5, label=label, zorder=5)

ax2.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax2.set_ylabel("Real Interest Rate (%)", fontsize=15)
ax2.tick_params(axis="y", labelsize=13)

# — X-axis formatting —
ax1.set_xlabel("Date", fontsize=15)
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax1.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax1.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax1.grid(axis="y", which="major", linestyle="--", linewidth=0.4, alpha=0.4)
plt.xticks(rotation=45, fontsize=13)

# — Combined legend —
lines  = [l for l in ax1.get_lines() + ax2.get_lines()
          if not l.get_label().startswith("_")]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12),
           fontsize=13, framealpha=0.85, ncol=2)

plt.title("Gold Price  vs  Real Interest Rates (DGS10 / DGS5 / Fed Funds − CPI YoY, Quarterly)",
          fontsize=16, pad=14)
ax1.set_xlim(df["Date"].iloc[0], df["Date"].iloc[-1])
plt.tight_layout()
plt.savefig("gold_realrate_chart.png", dpi=300, bbox_inches="tight")
plt.show()
print("Chart saved → gold_realrate_chart.png")
