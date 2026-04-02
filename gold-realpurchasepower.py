import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 1. Load data ───────────────────────────────────────────────────────────────
gold = pd.read_csv("monthly-gold-price.csv")
gold["Date"] = pd.to_datetime(gold["Date"])

cpi = pd.read_csv("CPIAUCSL.csv")
cpi = cpi.rename(columns={"observation_date": "Date", "CPIAUCSL": "CPI"})
cpi["Date"] = pd.to_datetime(cpi["Date"])

# ── 2. CPI base = Feb 2026 ─────────────────────────────────────────────────────
cpi_base = cpi.loc[cpi["Date"] == "2026-02-01", "CPI"].values[0]  # 327.460

# ── 3. Merge & calculate ───────────────────────────────────────────────────────
df = (gold.merge(cpi, on="Date")
          .sort_values("Date")
          .reset_index(drop=True))

df["Real_2026"] = df["Price"] * (cpi_base / df["CPI"])

# ── 4. Plot ────────────────────────────────────────────────────────────────────
COLOR_GOLD = "#D4A017"
COLOR_GREY = "#888888"

num_years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
fig, ax = plt.subplots(figsize=(max(20, num_years * 0.25), 9))

# Real (inflation-adjusted) price
ax.plot(df["Date"], df["Real_2026"],
        color=COLOR_GOLD, linewidth=1.2,
        label="Gold Price in 2026 Feb USD (inflation-adjusted)")
ax.fill_between(df["Date"], df["Real_2026"], 0,
                alpha=0.15, color=COLOR_GOLD)

# Nominal price
ax.plot(df["Date"], df["Price"],
        color=COLOR_GREY, linewidth=0.8, alpha=0.7,
        label="Nominal Gold Price (USD)")

# ── 5. Axes & formatting ───────────────────────────────────────────────────────
ax.set_ylabel("Gold Price (USD)", fontsize=16)
ax.set_xlabel("Date", fontsize=16)
ax.tick_params(axis="both", labelsize=14)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))
ax.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax.grid(axis="y", which="major", linestyle="--", linewidth=0.4, alpha=0.4)
plt.xticks(rotation=45)
ax.set_xlim(pd.Timestamp("1965-01-01"), df["Date"].iloc[-1])
ax.legend(fontsize=14, framealpha=0.85)

plt.title("Gold Price: Nominal vs Real (2026 Feb Purchasing Power)",
          fontsize=17, pad=14)
plt.tight_layout()
plt.savefig("gold_realpurchasepower_chart.png", dpi=300, bbox_inches="tight")
plt.show()
print("Chart saved → gold_realpurchasepower_chart.png")
