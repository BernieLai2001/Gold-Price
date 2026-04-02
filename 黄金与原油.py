import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import warnings, os
warnings.filterwarnings("ignore")

BASE = "/Users/laikaiyuan/Downloads/读书/黄金大师课/Gold Price/"

plt.rcParams["font.family"]     = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ── 读取数据 ─────────────────────────────────────────────
gold = pd.read_csv(BASE + "monthly-gold-price.csv", parse_dates=["Date"])
gold = gold.rename(columns={"Date": "date", "Price": "gold"})
gold["date"] = pd.to_datetime(gold["date"])

oil_raw = pd.read_csv(BASE + "from 1947 WTI crude oil prices.csv")
oil_raw.columns = ["date", "oil"]
oil_raw["date"] = pd.to_datetime(oil_raw["date"], format="%m/%d/%Y")

# 季度频率
gold_q = gold.set_index("date").resample("QS").mean()
oil_q  = oil_raw.set_index("date").resample("QS").mean()

real = pd.read_csv(BASE + "real_interest_rate.csv", parse_dates=["Date"])
real = real.rename(columns={"Date": "date"})[["date", "Real_DGS5"]]
real_q = real.set_index("date").resample("QS").mean()

df = gold_q.join(oil_q, how="inner").join(real_q, how="left")
df = df.dropna(subset=["gold", "oil"])
df["ratio"] = df["oil"] / df["gold"]   # 油价/金价比率

# 截取 1970 年之后
df = df[df.index >= "1970-01-01"]
dates = df.index
print(f"数据范围：{dates[0].date()} → {dates[-1].date()}，{len(dates)} 季度")

# ── 颜色 ─────────────────────────────────────────────────
C_GOLD  = "#C9912A"
C_OIL   = "#3D6B8C"
C_RATIO = "#7B3D7D"
C_REAL  = "#C0392B"

lw = 1.8

# ── 图形布局 2:1 上下 ────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# ── 上图：黄金（左轴） + 原油（右轴） ───────────────────
ax1r = ax1.twinx()

ax1.plot(dates, df["gold"], color=C_GOLD, lw=lw+0.3, label="黄金价格 (USD/oz)", zorder=4)
ax1r.plot(dates, df["oil"],  color=C_OIL,  lw=lw,     label="WTI原油 (USD/桶)", zorder=3)

ax1.fill_between(dates, df["gold"], alpha=0.12, color=C_GOLD, zorder=2)
ax1r.fill_between(dates, df["oil"],  alpha=0.10, color=C_OIL,  zorder=2)

# ── 下图：金油比 ─────────────────────────────────────────
ax2r = ax2.twinx()

ax2.plot(dates, df["ratio"], color=C_RATIO, lw=lw+0.3, label="油金比 (桶/盎司)", zorder=4)
ax2.fill_between(dates, df["ratio"], alpha=0.15, color=C_RATIO, zorder=2)
ax2.axhline(df["ratio"].mean(), color=C_RATIO, lw=1.2, ls="--", alpha=0.6)
ax2.text(dates[-1], df["ratio"].mean(),
         f" 均值 {df['ratio'].mean():.4f}",
         color=C_RATIO, fontsize=13, va="center")

# 实际利率（右轴），对0度线填充
real_vals = df["Real_DGS5"].values
ax2r.plot(dates, real_vals, color=C_REAL, lw=lw, label="DGS5实际利率 (%)", zorder=5)
ax2r.fill_between(dates, real_vals, 0,
                  where=(real_vals >= 0), color=C_REAL, alpha=0.15,
                  interpolate=True, zorder=3)
ax2r.fill_between(dates, real_vals, 0,
                  where=(real_vals < 0),  color=C_REAL, alpha=0.30,
                  interpolate=True, zorder=3)
ax2r.axhline(0, color=C_REAL, lw=1.0, ls=":", alpha=0.7)

# ── 重要事件 ─────────────────────────────────────────────
events = [
    ("1971-08-15", "尼克松\n关闭金窗",   "#B71C1C", "top"),
    ("1973-10-01", "第一次\n石油危机",    "#E65100", "top"),
    ("1979-01-01", "伊朗革命\n第二次油危","#E65100", "bot"),
    ("1985-11-01", "OPEC增产\n油价崩溃",  "#1565C0", "top"),
    ("1990-08-01", "海湾战争",            "#6A1B9A", "bot"),
    ("2001-09-11", "9·11",               "#333333", "top"),
    ("2003-03-01", "伊拉克战争",          "#6A1B9A", "bot"),
    ("2008-07-01", "油价峰值\n$147",       "#E65100", "top"),
    ("2008-09-01", "金融危机",            "#B71C1C", "bot"),
    ("2014-11-01", "OPEC不减产\n油价暴跌","#1565C0", "top"),
    ("2020-04-01", "疫情\n油价负数",       "#00695C", "bot"),
    ("2022-02-01", "俄乌战争",            "#B71C1C", "top"),
]

# ylim 先设好再画文字
g_min, g_max = df["gold"].min(), df["gold"].max()
g_rng = g_max - g_min
ax1.set_ylim(g_min - g_rng * 0.15, g_max + g_rng * 0.25)

o_min, o_max = df["oil"].min(), df["oil"].max()
o_rng = o_max - o_min
ax1r.set_ylim(o_min - o_rng * 0.15, o_max + o_rng * 0.25)

r_min, r_max = df["ratio"].min(), df["ratio"].max()
r_rng = r_max - r_min
ax2.set_ylim(r_min - r_rng * 0.15, r_max + r_rng * 0.30)

ri_min = df["Real_DGS5"].dropna().min()
ri_max = df["Real_DGS5"].dropna().max()
ri_rng = ri_max - ri_min
ax2r.set_ylim(ri_min - ri_rng * 0.15, ri_max + ri_rng * 0.30)

# 画竖线 + 文字（两图同步）
for date_str, label, color, pos in events:
    xd = pd.Timestamp(date_str)
    if xd < dates[0] or xd > dates[-1]:
        continue
    for ax in [ax1, ax2]:
        ax.axvline(xd, color=color, lw=1.1, ls="--", alpha=0.65, zorder=6)

    # 上图文字
    y_lo, y_hi = ax1.get_ylim()
    y_rng = y_hi - y_lo
    if pos == "top":
        y_txt = y_hi - y_rng * 0.04
        va = "top"
    else:
        y_txt = y_lo + y_rng * 0.04
        va = "bottom"
    ax1.text(xd, y_txt, label, color=color,
             fontsize=10, fontweight="bold", va=va, ha="center",
             rotation=0, clip_on=True, zorder=7,
             bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))

# ── X轴 & 格式 ───────────────────────────────────────────
for ax in [ax1, ax2]:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=15, rotation=0)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.xaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.4)
    ax.set_axisbelow(True)

plt.setp(ax1.get_xticklabels(), visible=False)  # 上图隐藏x轴标签

# ── 轴标签 ───────────────────────────────────────────────
ax1.set_ylabel("黄金价格 (USD/oz)",   fontsize=17, color=C_GOLD, labelpad=8)
ax1r.set_ylabel("WTI原油价格 (USD/桶)", fontsize=17, color=C_OIL,  labelpad=8)
ax2.set_ylabel("油金比 (桶/盎司)",    fontsize=17, color=C_RATIO, labelpad=8)
ax2r.set_ylabel("DGS5实际利率 (%)",   fontsize=17, color=C_REAL,  labelpad=8)

ax1.tick_params(axis="y",  labelsize=15, colors=C_GOLD)
ax1r.tick_params(axis="y", labelsize=15, colors=C_OIL)
ax2.tick_params(axis="y",  labelsize=15, colors=C_RATIO)
ax2r.tick_params(axis="y", labelsize=15, colors=C_REAL)

ax1.spines["left"].set_color(C_GOLD)
ax1r.spines["right"].set_color(C_OIL)
ax2.spines["left"].set_color(C_RATIO)
ax2r.spines["right"].set_color(C_REAL)

# ── 图例 ─────────────────────────────────────────────────
h1, l1 = ax1.get_legend_handles_labels()
h1r, l1r = ax1r.get_legend_handles_labels()
ax1.legend(h1 + h1r, l1 + l1r, loc="upper left",
           fontsize=14, frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
h2,  l2  = ax2.get_legend_handles_labels()
h2r, l2r = ax2r.get_legend_handles_labels()
ax2.legend(h2 + h2r, l2 + l2r, loc="upper left",
           fontsize=14, frameon=True, framealpha=0.9, edgecolor="#BBBBBB")

# ── 标题 ─────────────────────────────────────────────────
ax1.set_title("黄金价格 vs WTI原油价格", fontsize=16, color="#444", pad=8)
ax2.set_title("油金比（原油价格 / 黄金价格）vs DGS5实际利率", fontsize=16, color="#444", pad=8)
fig.suptitle("黄金与原油价格比较", fontsize=26, fontweight="bold", y=1.01)

fig.tight_layout()
out = BASE + "黄金与原油_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"✅ 已保存：{out}")
plt.show()
