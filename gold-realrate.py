import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 1. Load data ───────────────────────────────────────────────────────────────
gold = pd.read_csv("monthly-gold-price.csv")
gold["Date"] = pd.to_datetime(gold["Date"])

real_rate = pd.read_csv("real_interest_rate.csv")
real_rate["Date"] = pd.to_datetime(real_rate["Date"])

# ── 2. Resample gold price to quarterly (last monthly value of each quarter) ──
gold_q = (gold.set_index("Date")["Price"]
              .resample("QS").last()
              .reset_index()
              .rename(columns={"Price": "Gold_Price"}))

# ── 3. Merge ───────────────────────────────────────────────────────────────────
df = (gold_q.merge(real_rate[["Date", "Real_Rate"]], on="Date")
            .sort_values("Date")
            .reset_index(drop=True))

# ── 4. Plot ────────────────────────────────────────────────────────────────────
COLOR_GOLD = "#D4A017"
COLOR_BLUE = "#2166AC"

num_years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
width  = max(20, num_years * 0.55)
height = 9
fig, ax1 = plt.subplots(figsize=(width, height))

# — Left axis: Gold Price —
ax1.plot(df["Date"], df["Gold_Price"],
         color=COLOR_GOLD, linewidth=1.2, label="Gold Price (USD)")
ax1.fill_between(df["Date"], df["Gold_Price"], 0,
                 alpha=0.15, color=COLOR_GOLD)
ax1.set_ylabel("Gold Price (USD)", color=COLOR_GOLD, fontsize=12)
ax1.tick_params(axis="y", labelcolor=COLOR_GOLD)

# — Right axis: Real Interest Rate —
ax2 = ax1.twinx()
ax2.plot(df["Date"], df["Real_Rate"],
         color=COLOR_BLUE, linewidth=1.2, label="Real Interest Rate (%)")
ax2.fill_between(df["Date"], df["Real_Rate"], 0,
                 where=df["Real_Rate"] >= 0,
                 alpha=0.20, color=COLOR_BLUE, interpolate=True)
ax2.fill_between(df["Date"], df["Real_Rate"], 0,
                 where=df["Real_Rate"] < 0,
                 alpha=0.20, color=COLOR_BLUE, interpolate=True)
ax2.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax2.set_ylabel("Real Interest Rate (%)", color=COLOR_BLUE, fontsize=12)
ax2.tick_params(axis="y", labelcolor=COLOR_BLUE)

# — X-axis formatting —
ax1.set_xlabel("Date", fontsize=12)
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax1.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax1.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax1.grid(axis="y", which="major", linestyle="--", linewidth=0.4, alpha=0.4)
plt.xticks(rotation=45)

# — Combined legend —
lines  = [l for l in ax1.get_lines() + ax2.get_lines()
          if not l.get_label().startswith("_")]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left", fontsize=10, framealpha=0.85)

plt.title("Gold Price  vs  Real Interest Rate (DGS10 − CPI YoY, Quarterly)",
          fontsize=13, pad=12)
ax1.set_xlim(df["Date"].iloc[0], df["Date"].iloc[-1])
plt.tight_layout()
plt.savefig("gold_realrate_chart.png", dpi=300, bbox_inches="tight")
plt.show()
print("Chart saved → gold_realrate_chart.png")
