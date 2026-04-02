import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

plt.rcParams["font.family"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── 1. 加载数据 ────────────────────────────────────────────────────────────────
tax = pd.read_csv("联邦政府现行税收收据.csv")
tax.columns = ["Date", "Tax"]
tax["Date"] = pd.to_datetime(tax["Date"])
tax["Tax"] = pd.to_numeric(tax["Tax"], errors="coerce")

interest = pd.read_csv("联邦政府经常性支出：利息支付.csv")
interest.columns = ["Date", "Interest"]
interest["Date"] = pd.to_datetime(interest["Date"])
interest["Interest"] = pd.to_numeric(interest["Interest"], errors="coerce")

gdp = pd.read_csv("1947-present GDP.csv")
gdp.columns = ["Date", "GDP"]
gdp["Date"] = pd.to_datetime(gdp["Date"])
gdp["GDP"] = pd.to_numeric(gdp["GDP"], errors="coerce")

# ── 2. 季度化（已是季度，直接对齐到 QS） ─────────────────────────────────────
tax_q      = tax.set_index("Date")["Tax"].resample("QS").last().reset_index()
interest_q = interest.set_index("Date")["Interest"].resample("QS").last().reset_index()
gdp_q      = gdp.set_index("Date")["GDP"].resample("QS").last().reset_index()

# ── 3. 合并并计算三个指标 ──────────────────────────────────────────────────────
df = tax_q.merge(interest_q, on="Date", how="inner")
df = df.merge(gdp_q, on="Date", how="left")
df = df.dropna(subset=["Tax", "Interest", "GDP"]).reset_index(drop=True)

df["InterestOverTax"] = df["Interest"] / df["Tax"] * 100    # 利息/税收 (%)
df["TaxOverGDP"]      = df["Tax"]      / df["GDP"] * 100    # 税收/GDP (%)
df["InterestOverGDP"] = df["Interest"] / df["GDP"] * 100    # 利息/GDP (%)

# 从 2000 年开始
df = df[df["Date"] >= "2000-01-01"].reset_index(drop=True)

# ── 4. 绘图 ────────────────────────────────────────────────────────────────────
COLOR_IT  = "#C62828"   # 深红：利息/税收
COLOR_TG  = "#1565C0"   # 深蓝：税收/GDP
COLOR_IG  = "#E65100"   # 橙色：利息/GDP

start = df["Date"].iloc[0]
end   = df["Date"].iloc[-1]
num_years = (end - start).days / 365.25
fig_w = max(20, num_years * 0.55)

fig, ax1 = plt.subplots(figsize=(fig_w, 11))
ax2 = ax1.twinx()

# 左轴：税收/GDP 和 利息/GDP（共用同一 % 尺度）
ax1.plot(df["Date"], df["TaxOverGDP"],
         color=COLOR_TG, linewidth=2.0, label="税收/GDP (%)", zorder=4)
ax1.fill_between(df["Date"], df["TaxOverGDP"], 0,
                 color=COLOR_TG, alpha=0.10, interpolate=True)

ax1.plot(df["Date"], df["InterestOverGDP"],
         color=COLOR_IG, linewidth=2.0, linestyle="--", label="利息/GDP (%)", zorder=4)
ax1.fill_between(df["Date"], df["InterestOverGDP"], 0,
                 color=COLOR_IG, alpha=0.10, interpolate=True)

_gdp_max = max(df["TaxOverGDP"].max(), df["InterestOverGDP"].max())
ax1.set_ylim(0, _gdp_max * 1.5)
ax1.set_ylabel("占GDP比例 (%)", fontsize=18)
ax1.tick_params(axis="y", labelsize=14)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

# 右轴：利息/税收（独立尺度，更能反映偿债压力）
ax2.plot(df["Date"], df["InterestOverTax"],
         color=COLOR_IT, linewidth=2.2, linestyle="-.",
         label="利息/税收 (%)", zorder=5)
ax2.fill_between(df["Date"], df["InterestOverTax"], 0,
                 color=COLOR_IT, alpha=0.12, interpolate=True)

_it_max = df["InterestOverTax"].max()
ax2.set_ylim(0, _it_max * 1.5)
ax2.set_ylabel("利息/税收 (%)", color=COLOR_IT, fontsize=18)
ax2.tick_params(axis="y", labelcolor=COLOR_IT, labelsize=14)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# ── 5. 重要事件竖线 ────────────────────────────────────────────────────────────
events = {
    "1980 沃尔克加息":   "1980-01-01",
    "2001 互联网泡沫":   "2001-03-01",
    "2008 金融危机":     "2008-09-15",
    "2020 新冠QE":       "2020-03-01",
    "2022 加息开始":     "2022-03-01",
}
for label, date in events.items():
    ts = pd.Timestamp(date)
    if start <= ts <= end:
        ax1.axvline(ts, color="gray", linewidth=1.2, linestyle=":", alpha=0.7, zorder=3)
        ax1.text(ts, _gdp_max * 1.42, label,
                 rotation=90, fontsize=10, color="gray",
                 va="top", ha="right", alpha=0.8)

# ── 6. x 轴格式 ───────────────────────────────────────────────────────────────
ax1.set_xlabel("Date", fontsize=18)
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.tick_params(axis="x", labelsize=15, rotation=45)
ax1.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax1.grid(axis="x", which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
ax1.set_xlim(start, end)

# ── 7. Legend ─────────────────────────────────────────────────────────────────
all_h, all_l = [], []
for ax in [ax1, ax2]:
    h, l = ax.get_legend_handles_labels()
    all_h += [x for x, lb in zip(h, l) if not lb.startswith("_")]
    all_l += [lb for lb in l if not lb.startswith("_")]
ax1.legend(all_h, all_l,
           loc="upper center", bbox_to_anchor=(0.5, -0.1),
           ncol=3, fontsize=18, framealpha=0.85)

plt.title("美国联邦政府利息负担：利息/税收 / 税收/GDP / 利息/GDP（2000–至今）",
          fontsize=22, pad=14)
plt.tight_layout()
plt.savefig("联邦利息负担.png", dpi=300, bbox_inches="tight")
plt.show()
print("Chart saved → 联邦利息负担.png")
