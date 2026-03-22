import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 1. Load data ──────────────────────────────────────────────────────────────
gold  = pd.read_csv("monthly-gold-price.csv")
cpi   = pd.read_csv("CPIAUCSL.csv")
dgs10 = pd.read_csv("DGS10.csv")

# ── 2. Standardise column names & parse dates ─────────────────────────────────
gold["Date"]  = pd.to_datetime(gold["Date"])
cpi           = cpi.rename(columns={"observation_date": "Date", "CPIAUCSL": "CPI"})
cpi["Date"]   = pd.to_datetime(cpi["Date"])
dgs10         = dgs10.rename(columns={"observation_date": "Date", "DGS10": "DGS10"})
dgs10["Date"] = pd.to_datetime(dgs10["Date"])

# DGS10 is daily → resample to month-start average
dgs10 = (dgs10.set_index("Date")
               .resample("MS").mean()
               .reset_index())

# ── 3. Merge ──────────────────────────────────────────────────────────────────
df = (gold.merge(cpi,   on="Date")
          .merge(dgs10, on="Date")
          .sort_values("Date")
          .reset_index(drop=True))

# ── 4. Core calculations ──────────────────────────────────────────────────────
# Real gold price = nominal gold / CPI index
df["Real_Gold"] = df["Price"] / df["CPI"]

# Resample to quarterly (last monthly value of each quarter)
df_q = (df.set_index("Date")[["Price", "Real_Gold", "CPI", "DGS10"]]
          .resample("QS").last()
          .reset_index())

# CPI YoY inflation rate on quarterly data (%) = same quarter vs. one year ago
df_q["Inflation"] = df_q["CPI"].pct_change(4, fill_method=None) * 100

# Real interest rate = 10-yr Treasury yield − CPI YoY inflation (both annualised)
df_q["Real_Rate"] = df_q["DGS10"] - df_q["Inflation"]

# Quarterly cumulative change in nominal gold price
df_q["dGold"] = df_q["Price"].diff()

df_q = df_q.dropna(subset=["dGold", "Real_Rate"])

# Truncate to 1970 onwards
df_q = df_q[df_q["Date"] >= "1970-01-01"].reset_index(drop=True)

# Save real interest rate to CSV
df_q[["Date", "DGS10", "Inflation", "Real_Rate"]].to_csv("real_interest_rate.csv", index=False)
print("Saved → real_interest_rate.csv")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
COLOR_GOLD = "#D4A017"
COLOR_BLUE = "#2166AC"

num_years = (df_q["Date"].iloc[-1] - df_q["Date"].iloc[0]).days / 365.25
width  = max(20, num_years * 0.55)
height = 9
fig, ax1 = plt.subplots(figsize=(width, height))

# — Left axis: quarterly cumulative change in nominal gold price (y-axis visually flipped) —
ax1.plot(df_q["Date"], df_q["dGold"],
         color=COLOR_GOLD, linewidth=1.0,
         label="Quarterly Δ Gold Price USD (y-axis inverted)")
ax1.fill_between(df_q["Date"], df_q["dGold"], 0,
                 where=df_q["dGold"] >= 0,
                 alpha=0.25, color=COLOR_GOLD, interpolate=True)
ax1.fill_between(df_q["Date"], df_q["dGold"], 0,
                 where=df_q["dGold"] < 0,
                 alpha=0.25, color=COLOR_GOLD, interpolate=True)
ax1.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax1.set_ylabel("Quarterly Δ Gold Price (USD, inverted)", color=COLOR_GOLD, fontsize=12)
ax1.tick_params(axis="y", labelcolor=COLOR_GOLD)

# — Right axis: real interest rate level —
ax2 = ax1.twinx()
ax2.plot(df_q["Date"], df_q["Real_Rate"],
         color=COLOR_BLUE, linewidth=1.0,
         label="Real Interest Rate (DGS10 − CPI YoY, %)")
ax2.fill_between(df_q["Date"], df_q["Real_Rate"], 0,
                 where=df_q["Real_Rate"] >= 0,
                 alpha=0.20, color=COLOR_BLUE, interpolate=True)
ax2.fill_between(df_q["Date"], df_q["Real_Rate"], 0,
                 where=df_q["Real_Rate"] < 0,
                 alpha=0.20, color=COLOR_BLUE, interpolate=True)
ax2.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax2.set_ylabel("Real Interest Rate (%)", color=COLOR_BLUE, fontsize=12)
ax2.tick_params(axis="y", labelcolor=COLOR_BLUE)

ax1.invert_yaxis()   # visually flip the real gold price axis (水平旋转)

# — 1998 reference line —
ax1.axvline(pd.Timestamp("1998-01-01"), color="red", linewidth=1.2,
            linestyle="--", alpha=0.8, label="1998-01-01")

# — X-axis formatting —
ax1.set_xlabel("Date", fontsize=12)
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # quarterly minor ticks
ax1.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax1.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax1.grid(axis="y", which="major", linestyle="--", linewidth=0.4, alpha=0.4)
plt.xticks(rotation=45)

# — Combined legend —
lines  = [l for l in ax1.get_lines() + ax2.get_lines()
          if not l.get_label().startswith("_")]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left", fontsize=10, framealpha=0.85)

plt.title("Quarterly Δ Gold Price (y-inverted)  vs  Real Interest Rate (DGS10 − CPI YoY, Quarterly)",
          fontsize=13, pad=12)
ax1.set_xlim(df_q["Date"].iloc[0], df_q["Date"].iloc[-1])
plt.tight_layout()
plt.savefig("real_gold_rate_chart.png", dpi=300, bbox_inches="tight")
plt.show()
print("Chart saved → real_gold_rate_chart.png")
