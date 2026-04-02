"""
金银比（Gold/Silver Ratio）与美债长短端利差（T10Y3M）+ 滚动相关性
数据来源：yfinance GC=F / SI=F，FRED T10Y3M
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BASE = "/Users/laikaiyuan/Downloads/读书/黄金大师课/Gold Price/"

# ════════════════════════════════════════════════════════
# 1. 下载数据
# ════════════════════════════════════════════════════════
def load_investing(filepath, col_name):
    raw = pd.read_csv(filepath, quotechar='"', skipinitialspace=True)
    raw.columns = raw.columns.str.strip().str.replace('"','')
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw[col_name] = pd.to_numeric(raw["Price"].astype(str).str.replace(",",""), errors="coerce")
    df = raw[["Date", col_name]].dropna().sort_values("Date").reset_index(drop=True)
    return df.set_index("Date").resample("D").ffill().reset_index()

print("读取 XAU_USD weekly 黄金...")
gc = load_investing(BASE + "XAU_USD weekly Historical Data.csv", "Gold")

print("读取 XAG_USD weekly 白银...")
si = load_investing(BASE + "XAG_USD weekly Historical Data.csv", "Silver")

print("读取 T10Y3M 利差...")
spr = pd.read_csv(BASE + "10年期固定到期国债减去3个月固定到期国债.csv")
spr.columns = ["Date","Spread"]
spr["Date"]   = pd.to_datetime(spr["Date"], errors="coerce")
spr["Spread"] = pd.to_numeric(spr["Spread"], errors="coerce")

# ════════════════════════════════════════════════════════
# 2. 合并 & 计算
# ════════════════════════════════════════════════════════
df = gc.merge(si, on="Date", how="inner")
df = df.merge(spr, on="Date", how="left")
df = df.dropna(subset=["Gold","Silver"]).sort_values("Date").reset_index(drop=True)
df["GSR"] = df["Gold"] / df["Silver"]

# 月频重采样后计算滚动相关性（避免前向填充造成虚假相关）
df_m = df.set_index("Date")[["GSR","Spread"]].resample("W").last().resample("MS").mean().dropna()
from scipy.stats import spearmanr

def rolling_spearman(s1, s2, window):
    result = [np.nan] * len(s1)
    arr1, arr2 = s1.values, s2.values
    for i in range(window - 1, len(s1)):
        a = arr1[i-window+1:i+1]
        b = arr2[i-window+1:i+1]
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() >= window // 2:
            result[i] = spearmanr(a[mask], b[mask])[0]
    return pd.Series(result, index=s1.index)

df_m["Corr36"] = rolling_spearman(df_m["GSR"], df_m["Spread"], 36)
df_m["Corr12"] = rolling_spearman(df_m["GSR"], df_m["Spread"], 12)
df_m = df_m.reset_index()

print(f"数据范围：{df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}，{len(df)} 行")
print(f"GSR 范围：{df['GSR'].min():.1f} – {df['GSR'].max():.1f}")

# ════════════════════════════════════════════════════════
# 3. 关键事件
# ════════════════════════════════════════════════════════
events = [
    ("2008-09-15", "金融危机",   "#C62828", "top"),
    ("2011-09-06", "黄金峰值",   "#E65100", "top"),
    ("2015-12-16", "Fed加息",    "#1565C0", "bottom"),
    ("2020-03-18", "COVID",      "#2E7D32", "top"),
    ("2022-02-24", "俄乌冲突",   "#6A1B9A", "bottom"),
    ("2023-03-10", "硅谷银行",   "#AD1457", "top"),
]

# ════════════════════════════════════════════════════════
# 4. 绘图布局：上图（双轴）+ 下图（相关性）
# ════════════════════════════════════════════════════════
num_years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
width = max(22, num_years * 0.85)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, 18),
                                gridspec_kw={"hspace": 0.06, "height_ratios": [3, 2]})
fig.patch.set_facecolor("#FAFAF8")
for ax in [ax1, ax2]:
    ax.set_facecolor("#FAFAF8")

dates = df["Date"]

C_GSR    = "#B8860B"   # 暗金   – 金银比
C_SPREAD = "#1565C0"   # 蓝     – 利差
C_INV    = "#C62828"   # 红     – 利差倒挂
C_C252   = "#6A1B9A"   # 深紫   – 252天相关
C_C126   = "#00838F"   # 青     – 126天相关
lw = 1.8

# ── 上图：金银比（左轴）+ 利差（右轴）────────────────────────────────────
ax1.plot(dates, df["GSR"], color=C_GSR, lw=lw+0.3, zorder=4,
         label="金银比 (Gold/Silver Ratio)")
ax1.fill_between(dates, df["GSR"], df["GSR"].min()*0.95,
                 color=C_GSR, alpha=0.10, linewidth=0)

ax1r = ax1.twinx()
spr_data = df["Spread"]
ax1r.plot(dates, spr_data, color=C_SPREAD, lw=lw, zorder=3,
          label="10Y-3M 利差 (%)")
ax1r.fill_between(dates, spr_data, 0,
                  where=(spr_data >= 0), color=C_SPREAD, alpha=0.10,
                  linewidth=0, interpolate=True)
ax1r.fill_between(dates, spr_data, 0,
                  where=(spr_data <  0), color=C_INV, alpha=0.25,
                  linewidth=0, interpolate=True, label="倒挂区间")
ax1r.axhline(0, color="#AAAAAA", lw=0.8, ls=":", alpha=0.8, zorder=2)

# ── 下图：滚动相关性 ──────────────────────────────────────────────────────
c_dates = df_m["Date"]
corr36  = df_m["Corr36"]
corr12  = df_m["Corr12"]

ax2.plot(c_dates, corr36, color=C_C252, lw=lw,     label="滚动相关性（36月≈3年）", zorder=4)
ax2.plot(c_dates, corr12, color=C_C126, lw=lw-0.4, label="滚动相关性（12月≈1年）",
         ls="--", alpha=0.85, zorder=3)
ax2.fill_between(c_dates, corr36, 0,
                 where=(corr36 >= 0), color=C_C252, alpha=0.12,
                 linewidth=0, interpolate=True)
ax2.fill_between(c_dates, corr36, 0,
                 where=(corr36 <  0), color=C_C252, alpha=0.25,
                 linewidth=0, interpolate=True)
ax2.axhline( 0,    color="#888", lw=1.0, ls=":", alpha=0.8, zorder=5)
ax2.axhline( 0.5,  color="#AAA", lw=0.7, ls=":", alpha=0.5, zorder=5)
ax2.axhline(-0.5,  color="#AAA", lw=0.7, ls=":", alpha=0.5, zorder=5)
for val, label in [(0,"0"),(0.5,"+0.5"),(-0.5,"-0.5")]:
    ax2.annotate(label, xy=(c_dates.iloc[-1], val), xytext=(5,0),
                 textcoords="offset points",
                 fontsize=11 if val==0 else 10,
                 color="#888" if val==0 else "#AAA", va="center")
ax2.set_ylim(-1.05, 1.05)

# ── 事件线 ───────────────────────────────────────────────────────────────
for ax in [ax1, ax2]:
    for date_str, label, color, pos in events:
        ax.axvline(pd.Timestamp(date_str), color=color, lw=1.2, ls="--", alpha=0.7, zorder=6)

def draw_labels(ax, pos_override=None):
    y_lo, y_hi = ax.get_ylim()
    y_rng = y_hi - y_lo
    for date_str, label, color, pos in events:
        p = pos_override or pos
        y_txt = y_hi - y_rng*0.04 if p=="top" else y_lo + y_rng*0.04
        ax.text(pd.Timestamp(date_str), y_txt, label, color=color,
                fontsize=11, fontweight="bold",
                va="top" if p=="top" else "bottom",
                ha="center", rotation=0, clip_on=True, zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, ec="none"))

# ── x 轴格式 ─────────────────────────────────────────────────────────────
for ax in [ax1, ax2]:
    ax.set_xlim(dates.iloc[0], dates.iloc[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=13, rotation=0)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.xaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

# ── ylim 并画事件文字 ─────────────────────────────────────────────────────
gsr_min, gsr_max = df["GSR"].min(), df["GSR"].max()
gsr_rng = gsr_max - gsr_min
ax1.set_ylim(gsr_min - gsr_rng*0.05, gsr_max + gsr_rng*0.22)

sp_min = spr_data.dropna().min(); sp_max = spr_data.dropna().max()
sp_rng = sp_max - sp_min
ax1r.set_ylim(sp_min - sp_rng*0.10, sp_max + sp_rng*0.35)

draw_labels(ax1)
draw_labels(ax2)

# ── 轴标签 ───────────────────────────────────────────────────────────────
ax1.set_ylabel("金银比 (oz Gold / oz Silver)", fontsize=15, color=C_GSR, labelpad=8)
ax1.tick_params(axis="y", labelsize=13, colors=C_GSR)
ax1.spines["left"].set_color(C_GSR)

ax1r.set_ylabel("10Y-3M 利差 (%)", fontsize=15, color=C_SPREAD, labelpad=8)
ax1r.tick_params(axis="y", labelsize=13, colors=C_SPREAD)
ax1r.spines["right"].set_color(C_SPREAD)

ax2.set_ylabel("Pearson 相关系数", fontsize=15, color=C_C252, labelpad=8)
ax2.tick_params(axis="y", labelsize=13, colors=C_C252)
ax2.spines["left"].set_color(C_C252)

# ── 标题 & 图例 ───────────────────────────────────────────────────────────
fig.suptitle("金银比 与 美债长短端利差", fontsize=24, fontweight="bold", y=0.998)
ax1.set_title("金银比（左轴）· 10Y-3M 美债利差（右轴）",
              fontsize=15, color="#444", pad=6)
ax2.set_title("金银比 与 利差 的滚动 Spearman 相关系数（月频数据，窗口：36月 / 12月）",
              fontsize=15, color="#444", pad=6)

top_handles = [
    Line2D([0],[0], color=C_GSR,    lw=2.2,             label="金银比 (Gold/Silver Ratio)"),
    Line2D([0],[0], color=C_SPREAD, lw=2,               label="10Y-3M 利差 (%)"),
    Line2D([0],[0], color=C_INV,    lw=6, alpha=0.35,   label="利差倒挂（衰退信号）"),
]
bot_handles = [
    Line2D([0],[0], color=C_C252, lw=2,              label="36月滚动相关性（3年）"),
    Line2D([0],[0], color=C_C126, lw=1.5, ls="--",   label="12月滚动相关性（1年）"),
]
ax1.legend(handles=top_handles, fontsize=13, loc="upper left",
           frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
ax2.legend(handles=bot_handles, fontsize=13, loc="lower left",
           frameon=True, framealpha=0.9, edgecolor="#BBBBBB")

plt.tight_layout(rect=[0, 0, 0.97, 0.998])
out = BASE + "金银比与利差_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✅ 已保存：{out}")
plt.show()
