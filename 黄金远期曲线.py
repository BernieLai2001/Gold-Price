"""
黄金期货远期曲线（Forward Curve）历史演变
数据来源：Barchart GCZ00-GCZ28 日频 + yfinance GCZ25/26 + 快照文件
"""

import os, glob
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
# 1. 加载所有 GCZ 合约
# ════════════════════════════════════════════════════════
def expiry_from_year(y2):
    y = 2000 + y2 if y2 < 50 else 1900 + y2
    return pd.Timestamp(f"{y}-12-26")

def load_barchart(fp):
    try:
        raw = pd.read_csv(fp, skiprows=1, skipfooter=1,
                          engine="python", on_bad_lines="skip")
        raw.columns = raw.columns.str.strip().str.replace('"', '')
        dc = next((c for c in raw.columns if "Date" in c), None)
        cc = next((c for c in raw.columns if c.strip().lower() == "close"), None)
        if not dc or not cc:
            return None
        df = raw[[dc, cc]].copy()
        df.columns = ["Date", "Close"]
        df["Date"]  = pd.to_datetime(df["Date"],  errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df.dropna().sort_values("Date").reset_index(drop=True)
    except:
        return None

contracts = {}   # {expiry: DataFrame}

for fp in sorted(glob.glob(BASE + "GCZ*_Barchart_Interactive_Chart_Daily_*.csv")):
    name = os.path.basename(fp)[:5]
    try:
        y2 = int(name[3:])
    except:
        continue
    exp = expiry_from_year(y2)
    df  = load_barchart(fp)
    if df is not None and len(df) > 10:
        contracts[exp] = df

# yfinance 补 GCZ25 / GCZ26
for ticker, y2 in [("GCZ25.CMX", 25), ("GCZ26.CMX", 26)]:
    exp = expiry_from_year(y2)
    if exp in contracts:
        continue
    raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if len(raw) == 0:
        continue
    s = raw[("Close", ticker)].reset_index()
    s.columns = ["Date", "Close"]
    s["Date"] = pd.to_datetime(s["Date"]).dt.tz_localize(None)
    contracts[exp] = s.dropna().sort_values("Date").reset_index(drop=True)

print(f"共加载 {len(contracts)} 个合约")

# ════════════════════════════════════════════════════════
# 2. 关键历史日期 + 颜色
# ════════════════════════════════════════════════════════
key_dates = [
    ("2008-09-15", "金融危机 Lehman",       "#C62828"),
    ("2011-09-06", "黄金峰值 $1920",         "#E65100"),
    ("2015-12-16", "Fed 首次加息",           "#F9A825"),
    ("2020-03-18", "COVID 崩盘",             "#2E7D32"),
    ("2022-03-08", "俄乌冲突 黄金$2070",     "#1565C0"),
    ("2026-03-29", "当前 2026-03-29",        "#6A1B9A"),
]

# ════════════════════════════════════════════════════════
# 3. 对每个日期提取远期曲线
# ════════════════════════════════════════════════════════
def get_curve(snap_date_str):
    snap = pd.Timestamp(snap_date_str)
    pts  = []
    for exp, df in contracts.items():
        if exp <= snap:
            continue
        # 找最近交易日（容忍5天）
        near = df[df["Date"] <= snap + pd.Timedelta("5d")]
        near = near[near["Date"] >= snap - pd.Timedelta("5d")]
        if len(near) == 0:
            continue
        price = near.sort_values("Date").iloc[-1]["Close"]
        days  = (exp - snap).days
        pts.append({"Expiry": exp, "Days": days, "Price": price})
    if not pts:
        return None
    return pd.DataFrame(pts).sort_values("Days").reset_index(drop=True)

curves = {}
for date_str, label, color in key_dates:
    c = get_curve(date_str)
    if c is not None and len(c) >= 2:
        # 近月现货（最短合约作为参考）
        spot_ref = c["Price"].iloc[0]
        c["Rate"] = (c["Price"] / spot_ref - 1) * (360 / c["Days"]) * 100
        curves[date_str] = (c, label, color)
        print(f"{date_str} {label}: {len(c)} 个合约, "
              f"近月=${spot_ref:.0f}, 最远={c['Days'].max()}天")

# ════════════════════════════════════════════════════════
# 4. 绘图：上图=价格曲线，下图=隐含年化利率
# ════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14),
                                gridspec_kw={"hspace": 0.12, "height_ratios": [1, 1]})
fig.patch.set_facecolor("#FAFAF8")
for ax in [ax1, ax2]:
    ax.set_facecolor("#FAFAF8")

for date_str, (c, label, color) in curves.items():
    # 标准化为相对近月的比率（去除绝对价格差异）
    spot_ref = c["Price"].iloc[0]
    c["PriceNorm"] = (c["Price"] / spot_ref - 1) * 100  # % vs near-month

    ax1.plot(c["Days"], c["PriceNorm"], color=color, lw=2.2,
             marker="o", markersize=5, label=f"{label}")
    ax2.plot(c["Days"], c["Rate"],      color=color, lw=2.2,
             marker="o", markersize=5, label=f"{label}")

# 参考线
ax1.axhline(0, color="#888", lw=1.0, ls=":", alpha=0.7)
ax2.axhline(0, color="#888", lw=1.0, ls=":", alpha=0.7)

# X 轴（天数 → 年份标签）
for ax in [ax1, ax2]:
    ax.set_xlim(0, 365 * 6)
    ax.set_xticks([90, 180, 365, 365*2, 365*3, 365*4, 365*5])
    ax.set_xticklabels(["90天", "6个月", "1年", "2年", "3年", "4年", "5年"],
                       fontsize=13)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.xaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=13)

# 标注 90 天竖线
for ax in [ax1, ax2]:
    ax.axvline(90, color="#999", lw=1.0, ls="--", alpha=0.6)
    ax.text(90, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 3,
            "90天", fontsize=10, color="#999", ha="center", va="bottom")

# 轴标签
ax1.set_ylabel("相对近月溢价 (%)",    fontsize=15, labelpad=8)
ax2.set_ylabel("隐含年化利率 (%)",    fontsize=15, labelpad=8)
ax1.set_xlabel("距到期时间",          fontsize=14)
ax2.set_xlabel("距到期时间",          fontsize=14)

# 标题 & 图例
fig.suptitle("黄金期货远期曲线演变  Gold Forward Curve — Historical Comparison",
             fontsize=22, fontweight="bold", y=1.01)
ax1.set_title("各到期合约相对近月溢价（% vs 近月合约）",
              fontsize=14, color="#444", pad=8)
ax2.set_title("各到期合约隐含年化利率（% = 年化 GOFO）",
              fontsize=14, color="#444", pad=8)

ax1.legend(fontsize=13, frameon=True, framealpha=0.9,
           edgecolor="#BBBBBB", loc="upper left")
ax2.legend(fontsize=13, frameon=True, framealpha=0.9,
           edgecolor="#BBBBBB", loc="upper left")

plt.tight_layout()
out = BASE + "黄金远期曲线_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n✅ 已保存：{out}")
plt.show()
