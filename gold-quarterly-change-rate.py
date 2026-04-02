import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 1. Load data ───────────────────────────────────────────────────────────────
gold = pd.read_csv("monthly-gold-price.csv")
gold["Date"] = pd.to_datetime(gold["Date"])

cpi = pd.read_csv("CPIAUCSL.csv")
cpi = cpi.rename(columns={"observation_date": "Date", "CPIAUCSL": "CPI"})
cpi["Date"] = pd.to_datetime(cpi["Date"])

# ── 2. Merge & calculate real gold price ──────────────────────────────────────
df = gold.merge(cpi, on="Date").sort_values("Date").reset_index(drop=True)
df["Real_Gold"] = df["Price"] / df["CPI"]

# ── 3. Resample to quarterly (last value of each quarter) ────────────────────
df_q = (df.set_index("Date")[["Price", "Real_Gold"]]
          .resample("QS").last()
          .reset_index())

# ── 4. Truncate to 1970 onwards ───────────────────────────────────────────────
df_q = df_q[df_q["Date"] >= "1970-01-01"].reset_index(drop=True)

# ── 5. Quarter-over-quarter change rate (%) ───────────────────────────────────
df_q["dGold_pct"]     = df_q["Price"].pct_change() * 100
df_q["dRealGold_pct"] = df_q["Real_Gold"].pct_change() * 100
df_q = df_q.dropna(subset=["dGold_pct", "dRealGold_pct"])

# ── 6. Plot — 优化填充逻辑 ────────────────────────────────────────────────────
COLOR_GOLD = "#D4A017"
COLOR_TEAL = "#21918C"

num_years = (df_q["Date"].iloc[-1] - df_q["Date"].iloc[0]).days / 365.25
fig, ax = plt.subplots(figsize=(max(22, num_years * 0.55), 9))

# 1. 底层：名义涨幅 → 实际涨幅之间（通胀贡献），纯金色
ax.fill_between(df_q["Date"], df_q["dGold_pct"], df_q["dRealGold_pct"],
                color=COLOR_GOLD, alpha=0.3,
                edgecolor=COLOR_GOLD, linewidth=0.5,
                label="Inflation Contribution (Nominal − Real)")

# 2. 顶层：实际涨幅 → 0（真实购买力增减），纯青色
ax.fill_between(df_q["Date"], df_q["dRealGold_pct"], 0,
                color=COLOR_TEAL, alpha=0.4,
                edgecolor=COLOR_TEAL, linewidth=0.5,
                label="Real Growth / Decline")

# 3. 线条（置于填充层之上）
ax.plot(df_q["Date"], df_q["dGold_pct"],
        color=COLOR_GOLD, linewidth=1.5, zorder=3,
        label="Nominal Change Rate (%)")
ax.plot(df_q["Date"], df_q["dRealGold_pct"],
        color=COLOR_TEAL, linewidth=1.5, zorder=4,
        label="Real Change Rate (%)")

ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.6, zorder=5)
ax.set_ylabel("Quarterly Change Rate (%)", fontsize=13)
ax.tick_params(axis="y", labelsize=11)

# — X-axis formatting —
ax.set_xlabel("Date", fontsize=13)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax.grid(axis="y", which="major", linestyle="--", linewidth=0.4, alpha=0.4)
plt.xticks(rotation=45, fontsize=11)

# — Remove blank margins —
ax.set_xlim(df_q["Date"].iloc[0], df_q["Date"].iloc[-1])

# — Legend —
ax.legend(loc="upper left", fontsize=11, framealpha=0.85)

plt.title("Nominal vs Real Gold Price — Quarterly Change Rate (%)",
          fontsize=15, pad=14)
plt.tight_layout()
plt.savefig("gold_quarterly_changerate_chart.png", dpi=300, bbox_inches="tight")
plt.show()
print("Chart saved → gold_quarterly_changerate_chart.png")
