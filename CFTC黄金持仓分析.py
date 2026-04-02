"""
CFTC 黄金持仓分析（2008 – 至今）
数据来源：CFTC Legacy COT + XAU_USD weekly
面板布局：
  1. 黄金价格（右轴）+ 投机净多头（左轴）
  2. 未平仓合约 (Open Interest)
  3. 非商业多头 & 空头（投机方向）
  4. 商业净持仓（套保，通常为负）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BASE = "/Users/laikaiyuan/Downloads/读书/黄金大师课/Gold Price/"
START = pd.Timestamp("2008-01-01")

# ════════════════════════════════════════════════════════
# 1. 加载 CFTC 数据
# ════════════════════════════════════════════════════════
cftc = pd.read_csv(BASE + "CFTC黄金完整持仓数据.csv")
cftc["Date"] = pd.to_datetime(cftc["Date"])
cftc = cftc[cftc["Date"] >= START].sort_values("Date").reset_index(drop=True)

# 过滤掉混入的迷你合约行（OI 远低于正常水平，Comm_Net=0 是特征）
n_before = len(cftc)
cftc = cftc[cftc["OI"] >= 100_000].reset_index(drop=True)
print(f"过滤掉 {n_before - len(cftc)} 行异常数据（迷你合约/副产品）")

# ════════════════════════════════════════════════════════
# 2. 加载黄金价格（周频）
# ════════════════════════════════════════════════════════
raw = pd.read_csv(BASE + "XAU_USD weekly Historical Data.csv",
                  quotechar='"', skipinitialspace=True)
raw.columns = raw.columns.str.strip().str.replace('"', '')
raw["Date"]  = pd.to_datetime(raw["Date"], errors="coerce")
raw["Gold"]  = pd.to_numeric(raw["Price"].astype(str).str.replace(",", ""), errors="coerce")
gold = raw[["Date", "Gold"]].dropna().sort_values("Date").reset_index(drop=True)
gold = gold[gold["Date"] >= START]

# 对齐到 CFTC 日期（最近邻，容忍 ±7 天）
gold_idx = gold.set_index("Date")["Gold"]
def align_gold(date):
    window = gold_idx[gold_idx.index.to_series().between(date - pd.Timedelta("7d"),
                                                          date + pd.Timedelta("7d"))]
    return window.iloc[-1] if len(window) else np.nan

cftc["Gold"] = cftc["Date"].apply(align_gold)

print(f"CFTC rows: {len(cftc)}")
print(f"日期范围：{cftc['Date'].iloc[0].date()} → {cftc['Date'].iloc[-1].date()}")
print(f"最新 Net_Speculative: {cftc['Net_Speculative'].iloc[-1]:,}")

# ════════════════════════════════════════════════════════
# 3. 关键事件
# ════════════════════════════════════════════════════════
events = [
    ("2008-09-15", "金融危机",      "#C62828"),
    ("2012-07-26", "欧债危机",      "#E65100"),
    ("2013-04-04", "日本 QQE",      "#6D4C41"),
    ("2015-12-16", "Fed 首次加息",  "#1565C0"),
    ("2019-07-31", "Fed 降息",      "#00838F"),
    ("2020-03-18", "COVID",         "#2E7D32"),
    ("2022-02-24", "俄乌冲突",      "#6A1B9A"),
    ("2022-03-16", "Fed 激进加息",  "#4527A0"),
    ("2024-09-18", "Fed 降息",      "#00695C"),
]

# ════════════════════════════════════════════════════════
# 4. 颜色
# ════════════════════════════════════════════════════════
C_GOLD    = "#D4A017"
C_NET_POS = "#C62828"   # 净多头 > 0
C_NET_NEG = "#1565C0"   # 净多头 < 0
C_OI      = "#546E7A"
C_LONG    = "#2E7D32"
C_SHORT   = "#C62828"
C_COMM    = "#1565C0"
BG        = "#FAFAF8"

dates = cftc["Date"]
num_years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
width = max(22, num_years * 1.0)

# ════════════════════════════════════════════════════════
# 5. 绘图（4 面板）
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(width, 22),
                          gridspec_kw={"hspace": 0.06,
                                       "height_ratios": [3, 2, 2, 2]})
fig.patch.set_facecolor(BG)
for ax in axes:
    ax.set_facecolor(BG)

lw = 1.6

# ── Panel 1：投机净多头（左轴）+ 黄金价格（右轴）──────────────────────────
ax1 = axes[0]
net = cftc["Net_Speculative"]
ax1.fill_between(dates, net, 0,
                 where=(net >= 0), color=C_NET_POS, alpha=0.35,
                 linewidth=0, interpolate=True)
ax1.fill_between(dates, net, 0,
                 where=(net < 0),  color=C_NET_NEG, alpha=0.35,
                 linewidth=0, interpolate=True)
ax1.plot(dates, net, color=C_NET_POS, lw=lw - 0.3, zorder=4)
ax1.axhline(0, color="#AAAAAA", lw=0.8, ls=":", alpha=0.8, zorder=2)

ax1r = ax1.twinx()
ax1r.plot(dates, cftc["Gold"], color=C_GOLD, lw=lw + 0.4, zorder=5,
          label="黄金价格 (USD/oz)")
ax1r.fill_between(dates, cftc["Gold"], cftc["Gold"].min() * 0.95,
                  color=C_GOLD, alpha=0.08, linewidth=0)

ax1.set_ylabel("投机净多头（手数）", fontsize=17, color=C_NET_POS, labelpad=8)
ax1.tick_params(axis="y", labelsize=15, colors=C_NET_POS)
ax1.spines["left"].set_color(C_NET_POS)
ax1r.set_ylabel("黄金价格 (USD/oz)", fontsize=17, color=C_GOLD, labelpad=8)
ax1r.tick_params(axis="y", labelsize=15, colors=C_GOLD)
ax1r.spines["right"].set_color(C_GOLD)
ax1.set_title("投机净多头（左轴）· 黄金价格（右轴）", fontsize=17, color="#444", pad=6)

handles_p1 = [
    Line2D([0],[0], color=C_NET_POS, lw=2, label="投机净多头 (NonComm Net)"),
    Line2D([0],[0], color=C_GOLD,    lw=2, label="黄金价格"),
]
ax1.legend(handles=handles_p1, fontsize=15, loc="upper left",
           frameon=True, framealpha=0.9, edgecolor="#BBBBBB")

# ── Panel 2：未平仓合约 ─────────────────────────────────────────────────────
ax2 = axes[1]
ax2.fill_between(dates, cftc["OI"], 0,
                 color=C_OI, alpha=0.40, linewidth=0)
ax2.plot(dates, cftc["OI"], color=C_OI, lw=lw, zorder=4)
ax2.set_ylabel("未平仓合约（手数）", fontsize=17, color=C_OI, labelpad=8)
ax2.tick_params(axis="y", labelsize=15, colors=C_OI)
ax2.spines["left"].set_color(C_OI)
ax2.set_title("未平仓合约 (Open Interest)", fontsize=17, color="#444", pad=6)
ax2.legend(handles=[Line2D([0],[0], color=C_OI, lw=2, label="Open Interest")],
           fontsize=15, loc="upper left", frameon=True,
           framealpha=0.9, edgecolor="#BBBBBB")

# ── Panel 3：非商业多头 & 空头 ───────────────────────────────────────────────
ax3 = axes[2]
ax3.fill_between(dates, cftc["NonComm_Long"], 0,
                 color=C_LONG, alpha=0.20, linewidth=0)
ax3.fill_between(dates, cftc["NonComm_Short"], 0,
                 color=C_SHORT, alpha=0.20, linewidth=0)
ax3.plot(dates, cftc["NonComm_Long"],  color=C_LONG,  lw=lw, label="非商业多头 (Long)")
ax3.plot(dates, cftc["NonComm_Short"], color=C_SHORT, lw=lw, label="非商业空头 (Short)", ls="--")
ax3.set_ylabel("持仓量（手数）", fontsize=17, labelpad=8)
ax3.tick_params(axis="y", labelsize=15)
ax3.set_title("非商业持仓（投机方向）— 多头 vs 空头", fontsize=17, color="#444", pad=6)
ax3.legend(fontsize=15, loc="upper left", frameon=True,
           framealpha=0.9, edgecolor="#BBBBBB")

# ── Panel 4：商业净持仓 ──────────────────────────────────────────────────────
ax4 = axes[3]
comm = cftc["Comm_Net"]
ax4.fill_between(dates, comm, 0,
                 where=(comm >= 0), color=C_LONG,  alpha=0.35,
                 linewidth=0, interpolate=True)
ax4.fill_between(dates, comm, 0,
                 where=(comm < 0),  color=C_COMM,  alpha=0.35,
                 linewidth=0, interpolate=True)
ax4.plot(dates, comm, color=C_COMM, lw=lw, zorder=4)
ax4.axhline(0, color="#AAAAAA", lw=0.8, ls=":", alpha=0.8, zorder=2)
ax4.set_ylabel("商业净持仓（手数）", fontsize=17, color=C_COMM, labelpad=8)
ax4.tick_params(axis="y", labelsize=15, colors=C_COMM)
ax4.spines["left"].set_color(C_COMM)
ax4.set_title("商业持仓净头寸（套保方向，通常为负）", fontsize=17, color="#444", pad=6)
ax4.legend(handles=[Line2D([0],[0], color=C_COMM, lw=2, label="商业净持仓 (Comm Net)")],
           fontsize=15, loc="upper left", frameon=True,
           framealpha=0.9, edgecolor="#BBBBBB")

# ── 事件线 ────────────────────────────────────────────────────────────────────
for ax in axes:
    for date_str, label, color in events:
        ax.axvline(pd.Timestamp(date_str), color=color,
                   lw=1.2, ls="--", alpha=0.65, zorder=6)

# 事件文字只画在最上面和最下面的面板
# 对于时间相近的事件，交错高度以避免重叠
# offset_frac: 正数向上偏移，负数向下偏移（相对于基准位置）
EVENT_LABEL_OFFSET = {
    "欧债危机":     0.00,
    "日本 QQE":     0.12,   # 向上错开一级（与欧债危机相近）
    "俄乌冲突":     0.00,
    "Fed 激进加息": 0.12,   # 向上错开一级（与俄乌冲突相近）
}

def draw_event_labels(ax, pos="top"):
    y_lo, y_hi = ax.get_ylim()
    y_rng = y_hi - y_lo
    base_frac = 0.04
    for date_str, label, color in events:
        extra = EVENT_LABEL_OFFSET.get(label, 0.0)
        if pos == "top":
            y_txt = y_hi - y_rng * (base_frac + extra)
            va = "top"
        else:
            y_txt = y_lo + y_rng * (base_frac + extra)
            va = "bottom"
        ax.text(pd.Timestamp(date_str), y_txt, label,
                color=color, fontsize=13, fontweight="bold",
                va=va, ha="center", clip_on=True, zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          alpha=0.82, ec="none"))

# ── X 轴格式 ──────────────────────────────────────────────────────────────────
for ax in axes:
    ax.set_xlim(dates.iloc[0], dates.iloc[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=15, rotation=0)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.xaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

# ylim 留白后画事件文字
for ax in [axes[0], axes[1], axes[2], axes[3]]:
    lo, hi = ax.get_ylim()
    rng = hi - lo
    ax.set_ylim(lo - rng * 0.05, hi + rng * 0.22)

draw_event_labels(axes[0], "top")
draw_event_labels(axes[1], "top")
draw_event_labels(axes[2], "top")
draw_event_labels(axes[3], "top")

# ── 隐藏上方三个面板的 x 刻度标签 ──────────────────────────────────────────
for ax in axes[:3]:
    ax.tick_params(labelbottom=False)

# ── 大标题 ────────────────────────────────────────────────────────────────────
plt.tight_layout()
fig.suptitle("CFTC 黄金持仓分析（2008 – 至今）",
             fontsize=26, fontweight="bold", y=1.0)
plt.subplots_adjust(top=0.967)
out = BASE + "CFTC黄金持仓分析_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✅ 已保存：{out}")
plt.show()
