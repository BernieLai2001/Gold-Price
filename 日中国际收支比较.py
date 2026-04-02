import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.family"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]

BASE = "/Users/laikaiyuan/Downloads/读书/黄金大师课/Gold Price/"

# ── 1. 加载数据 ────────────────────────────────────────────────────────────────

# 日本 NIIP（亿日元 → 万亿日元）
jp_niip_raw = pd.read_excel(BASE + "日本NIIP (国际投资头寸净额).xls", sheet_name="IIP", header=None)
jp_niip_rows = jp_niip_raw.iloc[8:37, [1, 13]].copy()
jp_niip_rows.columns = ["YearLabel", "NIIP_BY"]
jp_niip_rows["Year"] = jp_niip_rows["YearLabel"].str.extract(r"End of (\d{4})").astype(float)
jp_niip = jp_niip_rows.dropna(subset=["Year", "NIIP_BY"]).copy()
jp_niip["Date"] = pd.to_datetime(jp_niip["Year"].astype(int).astype(str) + "-12-31")
jp_niip["NIIP_TY"] = jp_niip["NIIP_BY"].astype(float) / 1000   # 万亿日元

# 日本债务/GDP（%）
jp_debt = pd.read_csv(BASE + "日本中央政府债务总额（占GDP的百分比）.csv")
jp_debt.columns = ["Date", "DebtGDP"]
jp_debt["Date"] = pd.to_datetime(jp_debt["Date"])

# 日本 GDP（季度，亿日元）→ 年化为万亿日元
jp_gdp_q = pd.read_csv(BASE + "日本GDP.csv")
jp_gdp_q.columns = ["Date", "GDP_BY"]
jp_gdp_q["Date"] = pd.to_datetime(jp_gdp_q["Date"])
jp_gdp_a = jp_gdp_q.set_index("Date")["GDP_BY"].resample("YS").mean().reset_index()
jp_gdp_a["GDP_TY"] = jp_gdp_a["GDP_BY"] / 1000   # 万亿日元

# 初次收入（日本 & 中国，USD → 千亿美元）
inc_raw = pd.read_csv(BASE + "净初级收入（国际收支平衡表，现价美元）.csv", skiprows=4)
year_cols = [c for c in inc_raw.columns if str(c).isdigit()]

jp_inc = inc_raw[inc_raw["Country Code"] == "JPN"].iloc[0][year_cols].dropna()
jp_inc = pd.DataFrame({"Year": jp_inc.index.astype(int), "PrimaryIncome_BUSD": jp_inc.values.astype(float) / 1e9})
jp_inc["Date"] = pd.to_datetime(jp_inc["Year"].astype(str) + "-12-31")

cn_inc = inc_raw[inc_raw["Country Code"] == "CHN"].iloc[0][year_cols].dropna()
cn_inc = pd.DataFrame({"Year": cn_inc.index.astype(int), "PrimaryIncome_BUSD": cn_inc.values.astype(float) / 1e9})
cn_inc["Date"] = pd.to_datetime(cn_inc["Year"].astype(str) + "-12-31")

# 中国各部门杠杆率（CNBS，季度，% of GDP）
cnbs_raw = pd.read_excel(BASE + "CNBS中国杠杆率数据.xlsx", sheet_name="Data", header=None)
cnbs = cnbs_raw.iloc[1:].copy()   # 跳过第0行（英文列名）
cnbs.columns = ["Date", "Household", "NonFinCorp", "CentralGov", "LocalGov",
                "GovTotal", "NonFinSector", "FinAsset", "FinLiability"]
cnbs["Date"] = pd.to_datetime(cnbs["Date"], errors="coerce")
for col in ["Household", "NonFinCorp", "LocalGov", "FinLiability"]:
    cnbs[col] = pd.to_numeric(cnbs[col], errors="coerce")
cnbs = cnbs.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# 中国 NIIP（亿美元 → 十亿美元）
cn_niip_raw = pd.read_excel(BASE + "中国国际投资头寸净额.xlsx", sheet_name="Annual(USD)", header=None)
col_headers = cn_niip_raw.iloc[2, :]
years_cn = [str(c).replace("end-", "").strip() for c in col_headers[1:] if str(c).startswith("end-")]
niip_vals = cn_niip_raw.iloc[4, 1: len(years_cn) + 1].values
cn_niip = pd.DataFrame({"Year": [int(y) for y in years_cn],
                         "NIIP_BUSD": [float(v) / 10 for v in niip_vals]})
cn_niip["Date"] = pd.to_datetime(cn_niip["Year"].astype(str) + "-12-31")

# ── 2. 截取 1995 年之后 ────────────────────────────────────────────────────────
START = "1995-01-01"
jp_niip  = jp_niip[jp_niip["Date"] >= START].reset_index(drop=True)
jp_debt  = jp_debt[jp_debt["Date"] >= START].reset_index(drop=True)
jp_gdp_a = jp_gdp_a[jp_gdp_a["Date"] >= START].reset_index(drop=True)
jp_inc   = jp_inc[jp_inc["Date"] >= START].reset_index(drop=True)
cn_niip  = cn_niip[cn_niip["Date"] >= START].reset_index(drop=True)
cn_inc   = cn_inc[cn_inc["Date"] >= START].reset_index(drop=True)
cnbs     = cnbs[cnbs["Date"] >= START].reset_index(drop=True)

# 日本 GNI-GDP = 初次收入（USD）→ 换算为万亿日元需要汇率，直接用亿美元展示
# 计算日本 NIIP/GDP (%)：按年份 join（NIIP=Dec31，GDP=Jan1，年份相同）
jp_gdp_a["Year"] = jp_gdp_a["Date"].dt.year
jp_niip["Year"] = jp_niip["Date"].dt.year
jp_merge = jp_niip.merge(jp_gdp_a[["Year", "GDP_TY"]], on="Year", how="left")
jp_merge["NIIP_pctGDP"] = jp_merge["NIIP_TY"] / jp_merge["GDP_TY"] * 100

# ── 3. 颜色 ───────────────────────────────────────────────────────────────────
C_DEBT     = "#B71C1C"   # 深红：日本债务/GDP
C_JP_NIIP  = "#1565C0"   # 深蓝：日本 NIIP/GDP
C_JP_INC   = "#E65100"   # 橙：日本初次收入
C_CN_NIIP  = "#2E7D32"   # 深绿：中国 NIIP
C_CN_INC   = "#6A1B9A"   # 紫：中国初次收入
# 各部门杠杆率颜色
C_HH       = "#E65100"   # 橙：居民部门
C_CORP     = "#1565C0"   # 蓝：非金融企业
C_LGOV     = "#AD1457"   # 玫红：地方政府
C_FIN_L    = "#00838F"   # 青：金融部门负债方

# ── 4. 绘图 ───────────────────────────────────────────────────────────────────
fig, (ax_jp, ax_cn) = plt.subplots(1, 2, figsize=(26, 10), sharey=False)
fig.suptitle("日本 vs 中国：国际投资头寸 / 初次收入账户 / 债务负担", fontsize=28, fontweight="bold", y=1.01)

# ════════════════════════════════════
# 左图：日本（4指标）
#   左轴  : NIIP（万亿日元，绝对值）
#   右轴1 : 债务/GDP（%，柱状）
#   右轴2 : 海外收入差额/初次收入账户（十亿USD）
# ════════════════════════════════════
C_JP_NIIP_ABS = "#1565C0"   # 深蓝：NIIP 绝对值

ax_jp2 = ax_jp.twinx()                      # 右轴1：债务/GDP
ax_jp3 = ax_jp.twinx()                      # 右轴2：初次收入
ax_jp3.spines["right"].set_position(("outward", 85))

# 左轴：日本 NIIP（万亿日元）
ax_jp.plot(jp_niip["Date"], jp_niip["NIIP_TY"],
           color=C_JP_NIIP_ABS, linewidth=2.2, marker="o", markersize=4,
           label="日本对外净资产 / NIIP（万亿日元）", zorder=4)
ax_jp.fill_between(jp_niip["Date"], jp_niip["NIIP_TY"], 0,
                   color=C_JP_NIIP_ABS, alpha=0.13, interpolate=True)
ax_jp.axhline(0, color=C_JP_NIIP_ABS, linewidth=0.6, linestyle=":", alpha=0.5)
ax_jp.set_ylabel("NIIP（万亿日元）", color=C_JP_NIIP_ABS, fontsize=17)
ax_jp.tick_params(axis="y", labelcolor=C_JP_NIIP_ABS, labelsize=15)
niip_max = jp_niip["NIIP_TY"].max()
ax_jp.set_ylim(-niip_max * 0.15, niip_max * 1.5)

# 右轴1：债务/GDP（%，柱状）
ax_jp2.bar(jp_debt["Date"], jp_debt["DebtGDP"], width=pd.Timedelta(days=300),
           color=C_DEBT, alpha=0.30, label="日本中央政府债务/GDP (%)", zorder=2)
ax_jp2.set_ylabel("债务/GDP (%)", color=C_DEBT, fontsize=17)
ax_jp2.tick_params(axis="y", labelcolor=C_DEBT, labelsize=15)
ax_jp2.set_ylim(0, jp_debt["DebtGDP"].max() * 2.2)   # 数据在下半部分，不遮 NIIP

# 右轴2：海外收入差额 = GNI-GDP = 初次收入账户（十亿USD）
ax_jp3.plot(jp_inc["Date"], jp_inc["PrimaryIncome_BUSD"],
            color=C_JP_INC, linewidth=2.0, linestyle="--", marker="s", markersize=4,
            label="海外收入差额 (GNI-GDP) / 初次收入账户（十亿USD）", zorder=5)
ax_jp3.fill_between(jp_inc["Date"], jp_inc["PrimaryIncome_BUSD"], 0,
                    color=C_JP_INC, alpha=0.12, interpolate=True)
ax_jp3.axhline(0, color=C_JP_INC, linewidth=0.6, linestyle=":", alpha=0.5)
ax_jp3.set_ylabel("海外收入差额 / 初次收入（十亿USD）", color=C_JP_INC, fontsize=17)
ax_jp3.tick_params(axis="y", labelcolor=C_JP_INC, labelsize=15)
inc_max = jp_inc["PrimaryIncome_BUSD"].max()
ax_jp3.set_ylim(-inc_max * 0.3, inc_max * 1.8)

# 合并图例
h1, l1 = ax_jp.get_legend_handles_labels()
h2, l2 = ax_jp2.get_legend_handles_labels()
h3, l3 = ax_jp3.get_legend_handles_labels()
ax_jp.legend(h1+h2+h3, l1+l2+l3, loc="upper left", fontsize=14, framealpha=0.85)

ax_jp.set_title("日本：NIIP（绝对值）/ 债务/GDP / 海外收入差额 / 初次收入账户", fontsize=18, pad=12)
ax_jp.set_xlabel("年份", fontsize=17)
ax_jp.xaxis.set_major_locator(mdates.YearLocator(5))
ax_jp.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_jp.xaxis.set_minor_locator(mdates.YearLocator(1))
ax_jp.tick_params(axis="x", labelsize=15, rotation=45)
ax_jp.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax_jp.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
xlim_end = max(jp_niip["Date"].max(), jp_inc["Date"].max(), jp_debt["Date"].max())
ax_jp.set_xlim(pd.Timestamp(START), xlim_end)

# ════════════════════════════════════
# 右图：中国
#   左轴 : NIIP + 净初次收入（十亿USD）
#   右轴 : 各部门杠杆率（% of GDP）
# ════════════════════════════════════
ax_cn2 = ax_cn.twinx()   # 右轴：部门杠杆率

# 左轴：NIIP（实线+填充）
ax_cn.plot(cn_niip["Date"], cn_niip["NIIP_BUSD"],
           color=C_CN_NIIP, linewidth=2.2, marker="o", markersize=4,
           label="中国 NIIP (十亿 USD)", zorder=4)
ax_cn.fill_between(cn_niip["Date"], cn_niip["NIIP_BUSD"], 0,
                   color=C_CN_NIIP, alpha=0.15, interpolate=True)

# 左轴：净初次收入（虚线+填充）
ax_cn.plot(cn_inc["Date"], cn_inc["PrimaryIncome_BUSD"],
           color=C_CN_INC, linewidth=2.0, linestyle="--", marker="s", markersize=4,
           label="中国净初次收入 (GNI-GDP, 十亿USD)", zorder=4)
ax_cn.fill_between(cn_inc["Date"], cn_inc["PrimaryIncome_BUSD"], 0,
                   where=cn_inc["PrimaryIncome_BUSD"] >= 0,
                   color=C_CN_INC, alpha=0.12, interpolate=True)
ax_cn.fill_between(cn_inc["Date"], cn_inc["PrimaryIncome_BUSD"], 0,
                   where=cn_inc["PrimaryIncome_BUSD"] < 0,
                   color=C_CN_INC, alpha=0.12, interpolate=True)
ax_cn.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax_cn.set_ylabel("十亿 USD", fontsize=17)
ax_cn.tick_params(axis="y", labelsize=15)
cn_y_min = min(cn_niip["NIIP_BUSD"].min(), cn_inc["PrimaryIncome_BUSD"].min())
cn_y_max = max(cn_niip["NIIP_BUSD"].max(), cn_inc["PrimaryIncome_BUSD"].max())
cn_rng = cn_y_max - cn_y_min
ax_cn.set_ylim(cn_y_min - cn_rng * 0.15, cn_y_max + cn_rng * 0.2)

# 右轴：各部门杠杆率（居民 / 非金融企业 / 地方政府 / 金融负债方）
sectors = [
    ("Household",   C_HH,    "居民部门杠杆率 (% GDP)",       "-",   "o"),
    ("NonFinCorp",  C_CORP,  "非金融企业杠杆率 (% GDP)",     "--",  "s"),
    ("FinLiability",C_FIN_L, "金融部门负债方杠杆率 (% GDP)", ":",   "D"),
]
for col, color, label, ls, mk in sectors:
    d = cnbs.dropna(subset=[col])
    ax_cn2.plot(d["Date"], d[col], color=color, linewidth=1.6,
                linestyle=ls, marker=mk, markersize=3,
                label=label, zorder=5, alpha=0.85)

ax_cn2.set_ylabel("杠杆率 (% of GDP)", fontsize=17)
ax_cn2.tick_params(axis="y", labelsize=15)
lev_max = max(cnbs[["Household","NonFinCorp","FinLiability"]].max())
lev_min = 0
ax_cn2.set_ylim(lev_min, lev_max * 1.4)

# 合并图例（左轴在上，右轴在下）
h4, l4 = ax_cn.get_legend_handles_labels()
h5, l5 = ax_cn2.get_legend_handles_labels()
ax_cn.legend(h4+h5, l4+l5, loc="upper left", fontsize=14, framealpha=0.85)

ax_cn.set_title("中国：NIIP / 净初次收入（左轴） + 各部门杠杆率（右轴）", fontsize=18, pad=12)
ax_cn.set_xlabel("年份", fontsize=17)
ax_cn.xaxis.set_major_locator(mdates.YearLocator(5))
ax_cn.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_cn.xaxis.set_minor_locator(mdates.YearLocator(1))
ax_cn.tick_params(axis="x", labelsize=15, rotation=45)
ax_cn.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax_cn.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax_cn.set_xlim(pd.Timestamp("2003-01-01"), cn_niip["Date"].max())

# ── 5. 保存 ───────────────────────────────────────────────────────────────────
plt.tight_layout()
outfile = BASE + "日中国际收支比较.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()
print(f"Chart saved → {outfile}")
