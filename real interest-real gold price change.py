import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 1. 加载数据 ──────────────────────────────────────────────────────────────
gold  = pd.read_csv("monthly-gold-price.csv")
cpi   = pd.read_csv("CPIAUCSL.csv")
dgs10 = pd.read_csv("DGS10.csv")
dgs5  = pd.read_csv("DGS5.csv")
ff    = pd.read_csv("Federal Funds Effective Rate.csv")

# ── 2. 标准化列名与日期解析 ──────────────────────────────────────────────────
gold["Date"]  = pd.to_datetime(gold["Date"])
cpi           = cpi.rename(columns={"observation_date": "Date", "CPIAUCSL": "CPI"})
cpi["Date"]   = pd.to_datetime(cpi["Date"])
dgs10         = dgs10.rename(columns={"observation_date": "Date"})
dgs10["Date"] = pd.to_datetime(dgs10["Date"])
dgs5          = dgs5.rename(columns={"observation_date": "Date"})
dgs5["Date"]  = pd.to_datetime(dgs5["Date"])
ff            = ff.rename(columns={"observation_date": "Date", "DFF": "FF"})
ff["Date"]    = pd.to_datetime(ff["Date"])

def to_monthly(df, col):
    return (df.set_index("Date")[[col]]
              .apply(pd.to_numeric, errors="coerce")
              .resample("MS").mean()
              .reset_index())

dgs10_m = to_monthly(dgs10, "DGS10")
dgs5_m  = to_monthly(dgs5,  "DGS5")
ff_m    = to_monthly(ff,    "FF")

# ── 3. 合并所有序列 ──────────────────────────────────────────────────────────
df = (gold.merge(cpi,     on="Date")
          .merge(dgs10_m, on="Date", how="left")
          .merge(dgs5_m,  on="Date", how="left")
          .merge(ff_m,    on="Date", how="left")
          .sort_values("Date")
          .reset_index(drop=True))

# ── 4. 核心计算 ──────────────────────────────────────────────────────────────
df["Real_Gold"] = df["Price"] / df["CPI"]
df_q = (df.set_index("Date")[["Price", "Real_Gold", "CPI", "DGS10", "DGS5", "FF"]]
          .resample("QS").last()
          .reset_index())

df_q["Inflation"] = df_q["CPI"].pct_change(4, fill_method=None) * 100
df_q["Real_DGS10"] = df_q["DGS10"] - df_q["Inflation"]
df_q["Real_DGS5"]  = df_q["DGS5"]  - df_q["Inflation"]
df_q["Real_FF"]    = df_q["FF"]     - df_q["Inflation"]
df_q["dGold"] = df_q["Price"].diff()

df_q = df_q.dropna(subset=["dGold"])
df_q = df_q[df_q["Date"] >= "1970-01-01"].reset_index(drop=True)

# ── 5. 绘图 ──────────────────────────────────────────────────────────────────
COLOR_GOLD   = "#D4A017"
COLOR_DGS10  = "#1565C0"
COLOR_DGS5   = "#66BB6A"
COLOR_FF     = "#D32F2F"

num_years  = (df_q["Date"].iloc[-1] - df_q["Date"].iloc[0]).days / 365.25
xlim_start = df_q["Date"].iloc[0]
xlim_end   = df_q["Date"].iloc[-1]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                figsize=(max(20, num_years * 0.55), 12),
                                gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.0})

# ── 上图：季度金价变动（倒置）────────────────────────────────────────────────
ax1.plot(df_q["Date"], df_q["dGold"], color=COLOR_GOLD, linewidth=1.0,
         label="Quarterly Δ Gold Price (USD)")
ax1.fill_between(df_q["Date"], df_q["dGold"], 0, color=COLOR_GOLD, alpha=0.20)
ax1.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
ax1.invert_yaxis()
ax1.set_ylabel("Δ Gold Price (USD, inverted)", color=COLOR_GOLD, fontsize=14)
ax1.tick_params(axis="y", labelcolor=COLOR_GOLD, labelsize=12)
ax1.set_xlim(xlim_start, xlim_end)
ax1.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax1.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax1.grid(axis="y", which="major", linestyle="--", linewidth=0.4, alpha=0.4)
ax1.axvline(pd.Timestamp("1998-01-01"), color="red", linewidth=1.2, linestyle="--", alpha=0.7)
ax1.legend(loc="upper left", fontsize=12, framealpha=0.85)
ax1.set_title("Quarterly Δ Gold Price  vs  Real Interest Rates (DGS10 / DGS5 / Fed Funds − CPI YoY)",
              fontsize=15, pad=12)

# ── 下图：三条实际利率────────────────────────────────────────────────────────
fill_df = df_q.dropna(subset=["Real_DGS10", "Real_DGS5", "Real_FF"])
dates   = fill_df["Date"]
vals    = np.sort(np.stack([fill_df["Real_DGS10"].values,
                             fill_df["Real_DGS5"].values,
                             fill_df["Real_FF"].values], axis=1), axis=1)
bot, mid, top = vals[:, 0], vals[:, 1], vals[:, 2]

ax2.fill_between(dates, top, 0, color=COLOR_DGS10, alpha=0.20, linewidth=0)
ax2.fill_between(dates, mid, 0, color=COLOR_DGS5,  alpha=0.20, linewidth=0)
ax2.fill_between(dates, bot, 0, color=COLOR_FF,    alpha=0.20, linewidth=0)

for col, color, label in [
    ("Real_DGS10", COLOR_DGS10, "Real Rate — DGS10 (10Y)"),
    ("Real_DGS5",  COLOR_DGS5,  "Real Rate — DGS5 (5Y)"),
    ("Real_FF",    COLOR_FF,    "Real Rate — Fed Funds"),
]:
    ax2.plot(df_q["Date"], df_q[col], color=color, linewidth=1.5, label=label)

ax2.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.5)
ax2.axvline(pd.Timestamp("1998-01-01"), color="red", linewidth=1.2, linestyle="--", alpha=0.7)
ax2.set_ylabel("Real Interest Rate (%)", fontsize=14)
ax2.tick_params(axis="y", labelsize=12)
ax2.set_xlim(xlim_start, xlim_end)
ax2.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax2.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax2.grid(axis="y", which="major", linestyle="--", linewidth=0.4, alpha=0.4)
ax2.legend(loc="upper left", fontsize=12, framealpha=0.85, ncol=3)

# ── X 轴（shared，只在下图显示）─────────────────────────────────────────────
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax2.tick_params(axis="x", labelsize=12, rotation=45)
ax2.set_xlabel("Date", fontsize=14)

plt.tight_layout()
plt.savefig("real_gold_rate_change_chart.png", dpi=300, bbox_inches="tight")
plt.show()
print("Chart saved → real_gold_rate_change_chart.png")