import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.family"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]

# ── 1. 加载数据 ────────────────────────────────────────────────────────────────
debt = pd.read_csv("美国国债总额_19930401_20260326.csv",
                   usecols=["Record Date", "Total Public Debt Outstanding"])
debt = debt.rename(columns={"Record Date": "Date",
                             "Total Public Debt Outstanding": "Debt"})
debt["Date"] = pd.to_datetime(debt["Date"])
debt["Debt"] = debt["Debt"].astype(float) / 1e12          # 万亿美元

gdp = pd.read_csv("1947-present GDP.csv")
gdp = gdp.rename(columns={"observation_date": "Date"})
gdp["Date"] = pd.to_datetime(gdp["Date"])
gdp["GDP"]  = pd.to_numeric(gdp["GDP"], errors="coerce")  # 十亿美元

fed = pd.read_csv("美联储资产负债表.csv")
fed = fed.rename(columns={"observation_date": "Date", "WALCL": "FedAssets"})

treast = pd.read_csv("美联储直接持有的美国国债.csv")
treast = treast.rename(columns={"observation_date": "Date", "TREAST": "TREAST"})
treast["Date"] = pd.to_datetime(treast["Date"])
treast["TREAST"] = pd.to_numeric(treast["TREAST"], errors="coerce") / 1e6  # 万亿美元
fed["Date"]      = pd.to_datetime(fed["Date"])
fed["FedAssets"] = pd.to_numeric(fed["FedAssets"], errors="coerce") / 1e6  # 万亿美元

gold = pd.read_csv("monthly-gold-price.csv")
gold["Date"] = pd.to_datetime(gold["Date"])
gold = gold.sort_values("Date").reset_index(drop=True)

foreign = pd.read_csv("外国和国际投资者持有的联邦债务.csv")
foreign = foreign.rename(columns={"observation_date": "Date", "FDHBFIN": "ForeignHoldings"})
foreign["Date"]           = pd.to_datetime(foreign["Date"])
foreign["ForeignHoldings"] = pd.to_numeric(foreign["ForeignHoldings"], errors="coerce")  # 十亿美元

# 三大部门负债（百万美元）
fin_raw  = pd.read_csv("国内金融部门；债务证券和贷款；负债水平.csv")
fin_raw  = fin_raw.rename(columns={"observation_date": "Date", "DODFS": "FinDebt"})
fin_raw["Date"]    = pd.to_datetime(fin_raw["Date"])
fin_raw["FinDebt"] = pd.to_numeric(fin_raw["FinDebt"], errors="coerce")

corp_raw = pd.read_csv("非金融企业业务；债务证券和贷款；负债水平.csv")
corp_raw = corp_raw.rename(columns={"observation_date": "Date", "BCNSDODNS": "CorpDebt"})
corp_raw["Date"]     = pd.to_datetime(corp_raw["Date"])
corp_raw["CorpDebt"] = pd.to_numeric(corp_raw["CorpDebt"], errors="coerce")

hh_raw   = pd.read_csv("家庭和非营利组织负债水平.csv")
hh_raw   = hh_raw.rename(columns={"observation_date": "Date", "CMDEBT": "HHDebt"})
hh_raw["Date"]   = pd.to_datetime(hh_raw["Date"])
hh_raw["HHDebt"] = pd.to_numeric(hh_raw["HHDebt"], errors="coerce")

# ── 2. 季度化 ─────────────────────────────────────────────────────────────────
debt_q    = debt.set_index("Date")["Debt"].resample("QS").last().reset_index()
gdp_q     = gdp[["Date", "GDP"]].copy()   # GDP 已是季度数据
fed_q     = fed.set_index("Date")["FedAssets"].resample("QS").last().reset_index()
treast_q  = treast.set_index("Date")["TREAST"].resample("QS").last().reset_index()
foreign_q = foreign.set_index("Date")["ForeignHoldings"].resample("QS").last().reset_index()
fin_q     = fin_raw.set_index("Date")["FinDebt"].resample("QS").last().reset_index()
corp_q    = corp_raw.set_index("Date")["CorpDebt"].resample("QS").last().reset_index()
hh_q      = hh_raw.set_index("Date")["HHDebt"].resample("QS").last().reset_index()
gold_q    = gold.set_index("Date")["Price"].resample("QS").last().reset_index()

# ── 3. 计算各指标 ─────────────────────────────────────────────────────────────
dg = debt_q.merge(gdp_q, on="Date", how="left")
dg["DebtGDP"] = dg["Debt"] * 1000 / dg["GDP"] * 100      # 万亿→十亿 / 十亿 × 100 = %

# 外国持有 / 美国国债总额 (%)：ForeignHoldings(十亿) / Debt(万亿×1000) × 100
fg = foreign_q.merge(debt_q, on="Date", how="left")
fg["ForeignPct"] = fg["ForeignHoldings"] / (fg["Debt"] * 1000) * 100

# 美联储持有美债 / 美国国债总额 (%)：TREAST(万亿) / Debt(万亿) × 100
fed_q = fed_q.merge(treast_q, on="Date", how="left").merge(debt_q, on="Date", how="left")
fed_q["FedPct"] = fed_q["TREAST"] / fed_q["Debt"] * 100

# 三大部门负债/GDP (%)：负债(百万) / GDP(十亿) × 0.1
sec = fin_q.merge(corp_q, on="Date", how="outer").merge(hh_q, on="Date", how="outer")
sec = sec.merge(gdp_q, on="Date", how="left")
sec["FinGDP"]  = sec["FinDebt"]  / sec["GDP"] * 0.1
sec["CorpGDP"] = sec["CorpDebt"] / sec["GDP"] * 0.1
sec["HHGDP"]   = sec["HHDebt"]   / sec["GDP"] * 0.1

# ── 4. 截取 2000 年至今 ────────────────────────────────────────────────────────
START = "1990-01-01"
dg     = dg[dg["Date"] >= START].dropna(subset=["DebtGDP"]).reset_index(drop=True)
fed_q  = fed_q[fed_q["Date"] >= START].dropna(subset=["FedPct"]).reset_index(drop=True)
fg     = fg[fg["Date"] >= START].dropna(subset=["ForeignPct"]).reset_index(drop=True)
sec    = sec[sec["Date"] >= START].reset_index(drop=True)
gold_q = gold_q[gold_q["Date"] >= START].dropna(subset=["Price"]).reset_index(drop=True)

start = min(dg["Date"].iloc[0], fed_q["Date"].iloc[0])
end   = max(dg["Date"].iloc[-1], fed_q["Date"].iloc[-1])
num_years = (end - start).days / 365.25

# ── 5. 绘图 ───────────────────────────────────────────────────────────────────
COLOR_DEBT    = "#B71C1C"   # 深红：美债/GDP
COLOR_FED     = "#1565C0"   # 深蓝：美联储资产负债表
COLOR_FOREIGN = "#2E7D32"   # 深绿：外国持有占比
COLOR_FIN     = "#6A1B9A"   # 紫色：金融部门/GDP
COLOR_CORP    = "#E65100"   # 橙色：企业部门/GDP
COLOR_HH      = "#00838F"   # 青色：居民部门/GDP
COLOR_GOLD    = "#D4A017"   # 金色：黄金价格

fig, ax1 = plt.subplots(figsize=(max(20, num_years * 0.7), 15))
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 80))
ax4 = ax1.twinx()
ax4.spines["right"].set_visible(False)

# ── 左轴：所有 /GDP 比例（美债 + 三大部门）────────────────────────────────────
bar_width = pd.Timedelta(days=55)

# 美债/GDP — 柱状图
ax1.bar(dg["Date"], dg["DebtGDP"], width=bar_width,
        color=COLOR_DEBT, alpha=0.45, label="美国国债/GDP (%)", zorder=2)
ax1.plot(dg["Date"], dg["DebtGDP"],
         color=COLOR_DEBT, linewidth=1.4, zorder=3)

# 三大部门负债/GDP — 折线（无填充）
sec_cols = [
    ("FinGDP",  COLOR_FIN,  "金融部门负债/GDP (%)",  2.0, (0, (5, 2))),
    ("CorpGDP", COLOR_CORP, "企业部门负债/GDP (%)",  2.0, (0, (3, 1, 1, 1))),
    ("HHGDP",   COLOR_HH,   "居民部门负债/GDP (%)",  2.0, "--"),
]
for col, color, label, lw, ls in sec_cols:
    s = sec.dropna(subset=[col])
    ax1.plot(s["Date"], s[col], color=color, linewidth=lw,
             linestyle=ls, label=label, zorder=4)

# 左轴量程：覆盖所有 GDP 比值
all_gdp_max = max(
    dg["DebtGDP"].max(),
    sec[["FinGDP", "CorpGDP", "HHGDP"]].max().max()
)
ax1.set_ylim(0, all_gdp_max * 1.667)   # 数据占底部 60%
ax1.set_ylabel("占GDP比例 (%)", fontsize=18)
ax1.tick_params(axis="y", labelsize=14)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# ── 右轴：美联储持有占比 + 外国持有占比（共用同一 scale）─────────────────────
ax2.plot(fed_q["Date"], fed_q["FedPct"],
         color=COLOR_FED, linewidth=2.0, label="美联储持有美债/美国国债 (%)", zorder=5)
ax2.fill_between(fed_q["Date"], fed_q["FedPct"], 0,
                 color=COLOR_FED, alpha=0.12, interpolate=True)

ax2.plot(fg["Date"], fg["ForeignPct"],
         color=COLOR_FOREIGN, linewidth=2.0, linestyle=(0, (4, 1)),
         label="外国投资者持有/美国国债 (%)", zorder=5)
ax2.fill_between(fg["Date"], fg["ForeignPct"], 0,
                 color=COLOR_FOREIGN, alpha=0.10, interpolate=True)

_pct_max = max(fed_q["FedPct"].max(), fg["ForeignPct"].max())
ax2.set_ylim(0, _pct_max * 1.667)
ax2.set_ylabel("持有美债占比 (%)", fontsize=18)
ax2.tick_params(axis="y", labelsize=14)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# ax3 不再使用，隐藏
ax3.yaxis.set_visible(False)
ax3.spines["right"].set_visible(False)

# ── 黄金价格（独立轴，不显示 y 轴，推到图表上方）────────────────────────────
ax4.plot(gold_q["Date"], gold_q["Price"],
         color=COLOR_GOLD, linewidth=2.0, linestyle=(0,(3,1)),
         label="黄金价格 (USD/oz)", zorder=5)
_g_max = gold_q["Price"].max()
_g_min = gold_q["Price"].min()
_g_rng = _g_max - _g_min
# 底部留 1.5 倍数据范围 → 金价占顶部约 40%（3:2 分布）
ax4.set_ylim(_g_min - _g_rng * 1.5, _g_max * 1.05)
ax4.yaxis.set_visible(False)

# ── 重要事件竖线 ──────────────────────────────────────────────────────────────
events = {
    "2002 小布什美国梦": "2002-06-01",
    "2008 金融危机":     "2008-09-15",
    "2010 QE2":          "2010-11-01",
    "2020 新冠QE":       "2020-03-01",
    "2022 缩表开始":     "2022-06-01",
}
for label, date in events.items():
    ts = pd.Timestamp(date)
    if start <= ts <= end:
        ax1.axvline(ts, color="gray", linewidth=1.2, linestyle=":", alpha=0.7, zorder=5)
        ax1.text(ts, dg["DebtGDP"].max() * 1.28, label,
                 rotation=90, fontsize=10, color="gray",
                 va="top", ha="right", alpha=0.85)

# ── x 轴格式 ──────────────────────────────────────────────────────────────────
ax1.set_xlabel("Date", fontsize=18)
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.tick_params(axis="x", labelsize=15, rotation=45)
ax1.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax1.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax1.set_xlim(start, end)

# ── Legend ────────────────────────────────────────────────────────────────────
all_h, all_l = [], []
for ax in [ax1, ax2, ax3, ax4]:
    h, l = ax.get_legend_handles_labels()
    all_h += [x for x, lb in zip(h, l) if not lb.startswith("_")]
    all_l += [lb for lb in l if not lb.startswith("_")]
ax1.legend(all_h, all_l,
           loc="upper center", bbox_to_anchor=(0.5, -0.1),
           ncol=3, fontsize=16, framealpha=0.85)

plt.title("美国债务扩张全景：国债/GDP · 部门杠杆 · 美联储持有美债占比 · 外国持有占比 · 黄金价格（1990–至今）",
          fontsize=22, pad=14)
plt.tight_layout()

outfile = "美债与美联储资产负债表.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()
print(f"Chart saved → {outfile}")
