"""
黄金租借利率（Gold Lease Rate）
公式：Lease Rate = 无风险利率 - GOFO_90
  - 2018-07-02 前：3个月 USD LIBOR
  - 2018-07-02 起：SOFR 90日均值（SOFR90DAYAVG）
GOFO_90：90天常数期限插值，来源同 GOFO计算.py
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
# 1. 复用 GOFO_90 计算
# ════════════════════════════════════════════════════════
print("下载 GC=F 现货数据...")
gcf_raw = yf.download("GC=F", period="max", auto_adjust=True, progress=False)
gcf = gcf_raw[("Close","GC=F")].reset_index()
gcf.columns = ["Date","Spot"]
gcf["Date"] = pd.to_datetime(gcf["Date"]).dt.tz_localize(None)
gcf = gcf.dropna().sort_values("Date").reset_index(drop=True)

# 各季度合约到期日（26日近似）
MONTH_EXPIRY = {"GCF":1, "GCG":2, "GCH":3, "GCJ":4, "GCK":5, "GCM":6, "GCN":7, "GCQ":8, "GCU":9, "GCV":10, "GCX":11, "GCZ":12}

def expiry_for_prefix(pfx, y2):
    y = 2000 + y2 if y2 < 50 else 1900 + y2
    m = MONTH_EXPIRY.get(pfx, 12)
    return pd.Timestamp(f"{y}-{m:02d}-26")

def load_barchart(fp, name):
    try:
        raw = pd.read_csv(fp, skiprows=1, skipfooter=1, engine="python", on_bad_lines="skip")
        raw.columns = raw.columns.str.strip().str.replace('"','')
        dc = next((c for c in raw.columns if "Date" in c), None)
        cc = next((c for c in raw.columns if c.strip().lower() in ["close","last"]), None)
        if not dc or not cc: return None
        df = raw[[dc, cc]].copy()
        df.columns = ["Date","Close"]
        df["Date"]  = pd.to_datetime(df["Date"],  errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df.dropna().sort_values("Date").reset_index(drop=True)
    except:
        return None

contracts = {}
# 加载四季度合约：H(Mar) M(Jun) U(Sep) Z(Dec) G(Feb)
for prefix in ["GCF","GCG","GCH","GCJ","GCK","GCM","GCN","GCQ","GCU","GCV","GCX","GCZ"]:
    for fp in sorted(glob.glob(BASE + f"{prefix}*_Barchart_Interactive_Chart_Daily_*.csv")):
        name = os.path.basename(fp)[:5]
        pfx  = name[:3]
        try: y2 = int(name[3:])
        except: continue
        exp = expiry_for_prefix(pfx, y2)
        if exp in contracts: continue
        df = load_barchart(fp, name)
        if df is not None and len(df) > 10:
            contracts[exp] = df

# yfinance 补 GCZ25/26
for ticker, y2 in [("GCZ25.CMX",25),("GCZ26.CMX",26)]:
    exp = expiry_for_prefix("GCZ", y2)
    if exp in contracts: continue
    raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if len(raw) == 0: continue
    s = raw[("Close",ticker)].reset_index()
    s.columns = ["Date","Close"]
    s["Date"] = pd.to_datetime(s["Date"]).dt.tz_localize(None)
    contracts[exp] = s.dropna().sort_values("Date").reset_index(drop=True)

by_m = {}
for e in contracts: by_m[e.month] = by_m.get(e.month,0)+1
print(f"共加载 {len(contracts)} 个合约：" + ", ".join(f"M{m}={n}" for m,n in sorted(by_m.items())))

rows = []
for exp, df in contracts.items():
    for _, row in df.iterrows():
        rows.append({"Date":row["Date"],"Expiry":exp,"FutClose":row["Close"]})
fut_all = pd.DataFrame(rows).dropna().sort_values(["Date","Expiry"]).reset_index(drop=True)

GARBAGE_DAYS = 45
merged = gcf.merge(fut_all, on="Date", how="inner")
merged["DaysToExp"] = (merged["Expiry"] - merged["Date"]).dt.days
merged = merged[merged["DaysToExp"] >= GARBAGE_DAYS].sort_values(["Date","DaysToExp"]).reset_index(drop=True)

TARGET = 90.0
records = []
for date, grp in merged.groupby("Date"):
    spot = grp["Spot"].iloc[0]
    grp  = grp.sort_values("DaysToExp").reset_index(drop=True)
    below = grp[grp["DaysToExp"] <= TARGET]
    above = grp[grp["DaysToExp"] >  TARGET]
    if len(above) == 0: continue
    far_row = above.iloc[0]
    t_far, p_far = far_row["DaysToExp"], far_row["FutClose"]
    if len(below) > 0:
        nr = below.iloc[-1]
        t_near, p_near = nr["DaysToExp"], nr["FutClose"]
    else:
        t_near, p_near = 0.0, spot
    if t_far == t_near: continue
    p_90    = p_near + (p_far - p_near) * (TARGET - t_near) / (t_far - t_near)
    gofo_90 = (p_90 / spot - 1) * (360 / TARGET) * 100
    records.append({"Date": date, "Spot": spot, "GOFO_90": gofo_90})

gofo = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
gofo = gofo[gofo["GOFO_90"].abs() <= 30].copy()
print(f"GOFO_90 范围：{gofo['Date'].iloc[0].date()} → {gofo['Date'].iloc[-1].date()}, {len(gofo)} 行")

# ════════════════════════════════════════════════════════
# 2. 加载无风险利率：LIBOR 和 SOFR
# ════════════════════════════════════════════════════════
# -- 3M LIBOR（Macrotrends 格式，跳过前几行注释）
libor_raw = pd.read_csv(
    BASE + "3-month-libor-rate-historical-chart.csv",
    skiprows=15, header=0,
    names=["Date","LIBOR"],
    on_bad_lines="skip"
)
libor_raw["Date"]  = pd.to_datetime(libor_raw["Date"],  errors="coerce")
libor_raw["LIBOR"] = pd.to_numeric(libor_raw["LIBOR"], errors="coerce")
libor = libor_raw.dropna().sort_values("Date").reset_index(drop=True)
print(f"LIBOR 范围：{libor['Date'].iloc[0].date()} → {libor['Date'].iloc[-1].date()}, {len(libor)} 行")

# -- SOFR 90日均值（FRED 格式）
sofr_raw = pd.read_csv(BASE + "SOFR90DAYAVG.csv")
sofr_raw.columns = ["Date","SOFR"]
sofr_raw["Date"] = pd.to_datetime(sofr_raw["Date"],  errors="coerce")
sofr_raw["SOFR"] = pd.to_numeric(sofr_raw["SOFR"], errors="coerce")
sofr = sofr_raw.dropna().sort_values("Date").reset_index(drop=True)
print(f"SOFR   范围：{sofr['Date'].iloc[0].date()} → {sofr['Date'].iloc[-1].date()}, {len(sofr)} 行")

# ════════════════════════════════════════════════════════
# 3. 拼接无风险利率：SOFR 发布日之后用 SOFR，之前用 LIBOR
# ════════════════════════════════════════════════════════
SOFR_START = pd.Timestamp("2018-07-02")   # SOFR 90日均值首个发布日

libor_part = libor[libor["Date"] <  SOFR_START][["Date","LIBOR"]].rename(columns={"LIBOR":"RFR"})
sofr_part  = sofr [sofr ["Date"] >= SOFR_START][["Date","SOFR" ]].rename(columns={"SOFR" :"RFR"})
libor_part["Source"] = "LIBOR 3M"
sofr_part ["Source"] = "SOFR 90d"
rfr = pd.concat([libor_part, sofr_part]).sort_values("Date").reset_index(drop=True)
print(f"无风险利率拼接：{rfr['Date'].iloc[0].date()} → {rfr['Date'].iloc[-1].date()}, {len(rfr)} 行")

# ════════════════════════════════════════════════════════
# 4. 合并 GOFO + 无风险利率，计算 Lease Rate
# ════════════════════════════════════════════════════════
df = gofo.merge(rfr[["Date","RFR","Source"]], on="Date", how="inner")
df["LeaseRate"] = df["RFR"] - df["GOFO_90"]
df = df.sort_values("Date").reset_index(drop=True)
print(f"租借利率数据范围：{df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}, {len(df)} 行")

# 区分 LIBOR 和 SOFR 段
df_libor = df[df["Source"] == "LIBOR 3M"].copy()
df_sofr  = df[df["Source"] == "SOFR 90d"].copy()

# ════════════════════════════════════════════════════════
# 5. 绘图
# ════════════════════════════════════════════════════════
C_LEASE  = "#C62828"   # 租借利率
C_GOFO   = "#1565C0"   # GOFO_90
C_RFR    = "#2E7D32"   # 无风险利率
C_SPOT   = "#B8860B"   # 黄金现货
C_SOFR   = "#4A148C"   # SOFR段颜色
C_LIBOR  = "#00695C"   # LIBOR段颜色

fig, axes = plt.subplots(2, 1, figsize=(24, 14),
                          gridspec_kw={"hspace": 0.10, "height_ratios": [1, 1]})
fig.patch.set_facecolor("#FAFAF8")
ax_rate, ax_lease = axes
for ax in axes:
    ax.set_facecolor("#FAFAF8")

# ── 上图：GOFO_90 vs 无风险利率 ─────────────────────────────────────────
# LIBOR 段
ax_rate.plot(df_libor["Date"], df_libor["RFR"],
             color=C_LIBOR, lw=1.6, label="3M LIBOR（2018前）")
ax_rate.plot(df_libor["Date"], df_libor["GOFO_90"],
             color=C_GOFO, lw=1.4, ls="--", alpha=0.8)
# SOFR 段
ax_rate.plot(df_sofr["Date"], df_sofr["RFR"],
             color=C_SOFR, lw=1.6, label="SOFR 90日均值（2018后）")
ax_rate.plot(df_sofr["Date"], df_sofr["GOFO_90"],
             color=C_GOFO, lw=1.4, ls="--", alpha=0.8, label="GOFO_90（年化%）")
# 切换分割线
ax_rate.axvline(SOFR_START, color="#999", lw=1.2, ls=":", alpha=0.8)
ax_rate.text(SOFR_START, ax_rate.get_ylim()[1] if ax_rate.get_ylim()[1]!=1 else 6,
             "SOFR起", fontsize=10, color="#999", ha="left", va="top")
ax_rate.axhline(0, color="#888", lw=0.8, ls=":", alpha=0.6)
ax_rate.set_ylabel("利率 (%)", fontsize=15, labelpad=8)
ax_rate.tick_params(axis="y", labelsize=13)
ax_rate.legend(fontsize=12, loc="upper left",
               frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
ax_rate.set_title("无风险利率（LIBOR/SOFR）vs GOFO_90", fontsize=14, color="#444", pad=8)

# ── 下图：黄金租借利率 Lease Rate ────────────────────────────────────────
# LIBOR 段
ax_lease.plot(df_libor["Date"], df_libor["LeaseRate"],
              color=C_LEASE, lw=1.8, alpha=0.9)
# SOFR 段
ax_lease.plot(df_sofr["Date"], df_sofr["LeaseRate"],
              color=C_LEASE, lw=1.8, alpha=0.9, label="黄金租借利率（Lease Rate = 无风险利率 − GOFO）")
ax_lease.fill_between(df["Date"], df["LeaseRate"], 0,
                      where=(df["LeaseRate"] >= 0),
                      color=C_LEASE, alpha=0.18, linewidth=0, interpolate=True)
ax_lease.fill_between(df["Date"], df["LeaseRate"], 0,
                      where=(df["LeaseRate"] < 0),
                      color="#1565C0", alpha=0.25, linewidth=0, interpolate=True,
                      label="Lease Rate < 0（黄金比美元更贵借）")
ax_lease.axhline(0, color="#888", lw=1.0, ls=":", alpha=0.7)
ax_lease.annotate("0%", xy=(df["Date"].iloc[-1], 0),
                  xytext=(6, 0), textcoords="offset points",
                  fontsize=12, color="#888", va="center")
ax_lease.axvline(SOFR_START, color="#999", lw=1.2, ls=":", alpha=0.8)
ax_lease.set_ylabel("租借利率 (%)", fontsize=15, color=C_LEASE, labelpad=8)
ax_lease.tick_params(axis="y", labelsize=13, colors=C_LEASE)
ax_lease.spines["left"].set_color(C_LEASE)

# 右轴：黄金现货价格
ax_lease_r = ax_lease.twinx()
ax_lease_r.plot(df["Date"], df["Spot"], color=C_SPOT, lw=1.6,
                alpha=0.75, label="黄金现货 GC=F (USD/oz)", zorder=2)
ax_lease_r.set_ylabel("黄金价格 (USD/oz)", fontsize=15, color=C_SPOT, labelpad=8)
ax_lease_r.tick_params(axis="y", labelsize=13, colors=C_SPOT)
ax_lease_r.spines["right"].set_color(C_SPOT)

# 合并图例
handles1, labels1 = ax_lease.get_legend_handles_labels()
handles2, labels2 = ax_lease_r.get_legend_handles_labels()
ax_lease.legend(handles1 + handles2, labels1 + labels2,
                fontsize=12, loc="upper left",
                frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
ax_lease.set_title("黄金租借利率 = 无风险利率 − GOFO_90",
                   fontsize=14, color="#444", pad=8)

# ── 共享 X 轴 ────────────────────────────────────────────────────────────
x_start = df["Date"].iloc[0]  - pd.Timedelta("30d")
x_end   = df["Date"].iloc[-1] + pd.Timedelta("30d")
for ax in [ax_rate, ax_lease]:
    ax.set_xlim(x_start, x_end)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=13, rotation=0)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.xaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

plt.tight_layout()
fig.subplots_adjust(top=0.93)
fig.suptitle("黄金租借利率（Gold Lease Rate）", fontsize=24,
             fontweight="bold", y=0.98)
out = BASE + "黄金租借利率_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n✅ 已保存：{out}")
plt.show()
