"""
GOFO（黄金远期拆借利率）计算图
近月：GC=F（COMEX 连续合约，视为现货）
远月：GCZ00-GCZ28（12月到期合约，Barchart 日频数据 + yfinance 补GCZ25/26）
GOFO = (GCZ / GC=F - 1) × (360 / 距12月26日天数) × 100%（年化）
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
# 1. 加载 GC=F 现货
# ════════════════════════════════════════════════════════
print("下载 GC=F 现货数据...")
gcf_raw = yf.download("GC=F", period="max", auto_adjust=True, progress=False)
gcf = gcf_raw[("Close","GC=F")].reset_index()
gcf.columns = ["Date", "Spot"]
gcf["Date"] = pd.to_datetime(gcf["Date"]).dt.tz_localize(None)
gcf = gcf.dropna().sort_values("Date").reset_index(drop=True)
print(f"  GC=F: {gcf['Date'].iloc[0].date()} → {gcf['Date'].iloc[-1].date()}, {len(gcf)}行")

# ════════════════════════════════════════════════════════
# 2. 加载所有 GCZ 合约
# ════════════════════════════════════════════════════════
def expiry_from_year(year_2digit):
    """December 26 到期日（两位数年份）"""
    y = 2000 + year_2digit if year_2digit < 50 else 1900 + year_2digit
    return pd.Timestamp(f"{y}-12-26")

def load_barchart(filepath, contract_name):
    """解析 Barchart 日频/周频 CSV"""
    try:
        raw = pd.read_csv(filepath, skiprows=1, skipfooter=1,
                          engine="python", on_bad_lines="skip")
        # 标准化列名
        raw.columns = raw.columns.str.strip().str.replace('"','')
        # 找日期列（Date Time 或 Date）
        date_col = next((c for c in raw.columns
                         if "Date" in c or "date" in c), None)
        if date_col is None:
            return None
        # 找收盘价列
        close_col = next((c for c in raw.columns
                          if c.strip().lower() in ["close","last"]), None)
        if close_col is None:
            return None
        df = raw[[date_col, close_col]].copy()
        df.columns = ["Date", "Close"]
        df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna().sort_values("Date").reset_index(drop=True)
        df["Contract"] = contract_name
        return df
    except Exception as e:
        print(f"  ⚠️ 跳过 {contract_name}: {e}")
        return None

contracts = {}   # {expiry_date: DataFrame}

# 各季度合约到期日（统一用26日近似）
MONTH_EXPIRY = {
    "GCF": 1,   # January
    "GCG": 2,   # February
    "GCH": 3,   # March
    "GCJ": 4,   # April   ← 新增
    "GCK": 5,   # May     ← 新增
    "GCM": 6,   # June
    "GCN": 7,   # July
    "GCQ": 8,   # August
    "GCU": 9,   # September
    "GCV": 10,  # October
    "GCX": 11,  # November
    "GCZ": 12,  # December
}

def expiry_for_prefix(prefix, year_2digit):
    y = 2000 + year_2digit if year_2digit < 50 else 1900 + year_2digit
    m = MONTH_EXPIRY.get(prefix[:3], 12)
    return pd.Timestamp(f"{y}-{m:02d}-26")

def load_contracts_by_prefix(prefix, label):
    """通用：按前缀加载所有 Barchart 合约文件"""
    pattern = os.path.join(BASE, f"{prefix}*_Barchart_Interactive_Chart_Daily_*.csv")
    loaded = 0
    for fp in sorted(glob.glob(pattern)):
        name = os.path.basename(fp)[:5]       # e.g. "GCZ25"
        pfx  = name[:3]                        # "GCZ"
        year_str = name[3:]                    # "25"
        try:
            year_2d = int(year_str)
        except:
            continue
        exp = expiry_for_prefix(pfx, year_2d)
        if exp in contracts:
            continue
        df = load_barchart(fp, name)
        if df is not None and len(df) > 10:
            contracts[exp] = df
            loaded += 1
    print(f"  [{label}] 加载 {loaded} 个合约")

# ── 加载全部合约（H=Mar, M=Jun, N=Jul, Q=Aug, U=Sep, Z=Dec, G=Feb）────────
for prefix, label in [
    ("GCF", "Jan    F"),
    ("GCG", "Feb    G"),
    ("GCH", "March  H"),
    ("GCJ", "Apr    J"),
    ("GCK", "May    K"),
    ("GCM", "June   M"),
    ("GCN", "July   N"),
    ("GCQ", "Aug    Q"),
    ("GCU", "Sept   U"),
    ("GCV", "Oct    V"),
    ("GCX", "Nov    X"),
    ("GCZ", "Dec    Z"),
]:
    load_contracts_by_prefix(prefix, label)

# ── yfinance 补 GCZ25/26（若 Barchart 缺失）────────────────────────────
for ticker, year_2d in [("GCZ25.CMX", 25), ("GCZ26.CMX", 26)]:
    exp = expiry_for_prefix("GCZ", year_2d)
    if exp in contracts:
        continue
    print(f"  从 yfinance 下载 {ticker}...")
    raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if len(raw) == 0:
        continue
    s = raw[("Close", ticker)].reset_index()
    s.columns = ["Date", "Close"]
    s["Date"]     = pd.to_datetime(s["Date"]).dt.tz_localize(None)
    s["Contract"] = ticker[:5]
    s = s.dropna().sort_values("Date").reset_index(drop=True)
    contracts[exp] = s

by_month = {}
for e in contracts:
    by_month[e.month] = by_month.get(e.month, 0) + 1
print(f"\n共加载 {len(contracts)} 个合约：" +
      ", ".join(f"M{m}={n}" for m, n in sorted(by_month.items())))

# ════════════════════════════════════════════════════════
# 3. 合并：对每个 GC=F 交易日，找干净合约插值到 90 天
#    Case A：Near(≥30天, ≤90天) + Far(>90天) → 真正插值
#    Case B：无干净 Near → GC=F 为 t=0 锚点 + Far → 外推
#    垃圾区保护：DaysToExp < GARBAGE_DAYS 的合约全部排除
#    GCH (March) 天然约 90 天，取代 GCZ 近月进入交割区后的角色
# ════════════════════════════════════════════════════════
TARGET      = 90.0
GARBAGE_DAYS = 45   # Near < 45天 → 提前排除，避免交割压力在30-45天窗口污染价格

# 将所有合约数据做成大表：Date + Expiry → Close
rows = []
for exp, df in contracts.items():
    for _, row in df.iterrows():
        rows.append({"Date": row["Date"], "Expiry": exp, "FutClose": row["Close"]})
fut_all = pd.DataFrame(rows).dropna().sort_values(["Date","Expiry"]).reset_index(drop=True)

# 合并 GC=F，排除垃圾区合约
merged = gcf.merge(fut_all, on="Date", how="inner")
merged["DaysToExp"] = (merged["Expiry"] - merged["Date"]).dt.days
merged = merged[merged["DaysToExp"] >= GARBAGE_DAYS].copy()
merged = merged.sort_values(["Date","DaysToExp"]).reset_index(drop=True)

# ── 90天常数期限插值 ────────────────────────────────────────────────────
records = []
for date, grp in merged.groupby("Date"):
    spot = grp["Spot"].iloc[0]
    grp  = grp.sort_values("DaysToExp").reset_index(drop=True)
    below = grp[grp["DaysToExp"] <= TARGET]
    above = grp[grp["DaysToExp"] >  TARGET]
    if len(above) == 0:
        continue
    t_far, p_far = above.iloc[0]["DaysToExp"], above.iloc[0]["FutClose"]
    if len(below) > 0:
        # Case A：Near(30-90天) < 90 < Far，真正插值
        t_near, p_near = below.iloc[-1]["DaysToExp"], below.iloc[-1]["FutClose"]
        case = "A"
    else:
        # Case B：无干净合约 < 90天，GC=F 为 t=0 锚点
        # （包含 Near 进入垃圾区后退化为此情况）
        t_near, p_near = 0.0, spot
        case = "B"
    if t_far == t_near:
        continue
    p_90    = p_near + (p_far - p_near) * (TARGET - t_near) / (t_far - t_near)
    gofo_90 = (p_90 / spot - 1) * (360 / TARGET) * 100
    records.append({
        "Date": date, "Spot": spot,
        "t_near": t_near, "P_near": p_near,
        "t_far":  t_far,  "P_far":  p_far,
        "P_90": p_90, "GOFO_90": gofo_90,
        "Basis_pct": (p_90 / spot - 1) * 100,
        "Case": case,
    })

best = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
best = best[best["GOFO_90"].abs() <= 30].copy()

n_a = (best["Case"] == "A").sum()
n_b = (best["Case"] == "B").sum()
print(f"\nGOFO 数据范围：{best['Date'].iloc[0].date()} → {best['Date'].iloc[-1].date()}, {len(best)} 行")
print(f"  Case A（真正插值 Near<90<Far）：{n_a} 天")
print(f"  Case B（GCF 为锚点，含垃圾区）：{n_b} 天")
print(f"  Case B（GCF 为锚点，Near=0）  ：{n_b} 天")
print(best[["Date","Spot","t_near","t_far","P_90","GOFO_90","Case"]].tail(5).to_string(index=False))

# ════════════════════════════════════════════════════════
# 4. 绘图
# ════════════════════════════════════════════════════════
C_SPOT  = "#B8860B"
C_FUT   = "#6D4C41"
C_GOFO  = "#1565C0"
C_BASIS = "#6A1B9A"

dates  = best["Date"]
X_START = dates.iloc[0]  - pd.Timedelta("30d")
X_END   = dates.iloc[-1] + pd.Timedelta("30d")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14),
                                gridspec_kw={"hspace": 0.10, "height_ratios": [2, 1]})
fig.patch.set_facecolor("#FAFAF8")
for ax in [ax1, ax2]:
    ax.set_facecolor("#FAFAF8")

# ── 上图：现货 + 期货 + 基差柱 ───────────────────────────────────────────
ax1_r = ax1.twinx()
ax1_r.spines["right"].set_visible(True)

ax1.plot(dates, best["Spot"],     color=C_SPOT, lw=1.8, label="黄金现货 GC=F (USD/oz)", zorder=4)
ax1.plot(dates, best["P_90"], color=C_FUT,  lw=1.5, ls="--",
         label="黄金90天远期价 P_90 (USD/oz)", zorder=4)
ax1.fill_between(dates, best["Spot"], best["P_90"],
                 where=(best["P_90"] >= best["Spot"]),
                 color=C_FUT, alpha=0.12, linewidth=0, interpolate=True,
                 label="Contango（P_90 > 现货）")
ax1.fill_between(dates, best["Spot"], best["P_90"],
                 where=(best["P_90"] < best["Spot"]),
                 color="#C62828", alpha=0.22, linewidth=0, interpolate=True,
                 label="Backwardation（P_90 < 现货）")

bar_colors = [C_BASIS if v >= 0 else "#C62828" for v in best["Basis_pct"]]
ax1_r.bar(dates, best["Basis_pct"], width=3, color=bar_colors, alpha=0.40,
          label="基差 % (期货/现货−1)")
ax1_r.axhline(0, color="#888888", lw=0.8, ls=":", alpha=0.7)

# ── 下图：GOFO ────────────────────────────────────────────────────────────
ax2.plot(dates, best["GOFO_90"], color=C_GOFO, lw=1.8, label="GOFO 年化 (%)", zorder=4)
ax2.fill_between(dates, best["GOFO_90"], 0,
                 where=(best["GOFO_90"] >= 0),
                 color=C_GOFO, alpha=0.15, linewidth=0, interpolate=True)
ax2.fill_between(dates, best["GOFO_90"], 0,
                 where=(best["GOFO_90"] <  0),
                 color="#C62828", alpha=0.28, linewidth=0, interpolate=True)
ax2.axhline(0, color="#888888", lw=1.0, ls=":", alpha=0.8, zorder=5)
ax2.annotate("0%", xy=(dates.iloc[-1], 0),
             xytext=(6, 0), textcoords="offset points",
             fontsize=13, color="#666666", va="center")

# ── X 轴 ──────────────────────────────────────────────────────────────────
for ax in [ax1, ax2]:
    ax.set_xlim(X_START, X_END)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=13, rotation=0)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.xaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

# ── 轴标签 ────────────────────────────────────────────────────────────────
ax1.set_ylabel("黄金价格 (USD/oz)",  fontsize=16, color=C_SPOT,  labelpad=8)
ax1_r.set_ylabel("基差 % (非年化)",   fontsize=16, color=C_BASIS, labelpad=8)
ax2.set_ylabel("GOFO 年化利率 (%)", fontsize=16, color=C_GOFO,  labelpad=8)
ax1.tick_params(axis="y",  labelsize=14, colors=C_SPOT)
ax1_r.tick_params(axis="y", labelsize=14, colors=C_BASIS)
ax2.tick_params(axis="y",  labelsize=14, colors=C_GOFO)
ax1.spines["left"].set_color(C_SPOT)
ax1_r.spines["right"].set_color(C_BASIS)
ax2.spines["left"].set_color(C_GOFO)

# ── 标题 & 图例 ───────────────────────────────────────────────────────────
fig.suptitle("黄金远期拆借利率 GOFO（GC=F 现货 vs 最近可用12月期货合约）",
             fontsize=22, fontweight="bold", y=1.01)
ax1.set_title("黄金现货 vs 12月期货价格（Contango / Backwardation）",
              fontsize=14, color="#444", pad=8)
ax2.set_title("GOFO-90 = 价格线性插值至90天后年化  (Case A: 真插值 / Case B: GCF锚点)",
              fontsize=14, color="#444", pad=8)

top_h = [
    Line2D([0],[0], color=C_SPOT,    lw=2,   label="黄金现货 GC=F"),
    Line2D([0],[0], color=C_FUT,     lw=2,   ls="--", label="90天远期价 P_90"),
    Line2D([0],[0], color=C_FUT,     lw=10,  alpha=0.3, label="Contango"),
    Line2D([0],[0], color="#C62828", lw=10,  alpha=0.4, label="Backwardation"),
    Line2D([0],[0], color=C_BASIS,   lw=10,  alpha=0.5, label="基差 % 柱"),
]
bot_h = [
    Line2D([0],[0], color=C_GOFO,    lw=2,   label="GOFO 90天常数期限 (%)"),
    Line2D([0],[0], color="#C62828", lw=8,   alpha=0.4, label="GOFO < 0 (Backwardation)"),
]
ax1.legend(handles=top_h, loc="upper left",
           fontsize=13, frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
ax2.legend(handles=bot_h, loc="upper left",
           fontsize=13, frameon=True, framealpha=0.9, edgecolor="#BBBBBB")

plt.tight_layout()
out = BASE + "GOFO计算_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n✅ 已保存：{out}")
plt.show()
