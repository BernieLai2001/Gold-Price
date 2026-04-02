import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

plt.rcParams["font.family"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]

# ── 1. 加载数据 ────────────────────────────────────────────────────────────────
gdp = pd.read_csv("1947-present GDP.csv")
gdp = gdp.rename(columns={"observation_date": "Date", gdp.columns[1]: "GDP"})
gdp["Date"] = pd.to_datetime(gdp["Date"])

corp = pd.read_csv("非金融企业业务；债务证券和贷款；负债水平.csv")
corp = corp.rename(columns={"observation_date": "Date", "BCNSDODNS": "CorpDebt"})
corp["Date"] = pd.to_datetime(corp["Date"])
corp["CorpDebt"] = pd.to_numeric(corp["CorpDebt"], errors="coerce")

# 美国家庭财富分配份额（联储 DFA，季度，1989 Q3–）
wealth_shares = pd.read_csv("美国家庭财富分配情况-shares.csv")
wealth_shares["Date"] = pd.PeriodIndex(
    wealth_shares["Date"].str.replace(":", ""), freq="Q"
).to_timestamp()
# 前 1%：TopPt1 + RemainingTop1 净资产份额相加
top1 = (wealth_shares[wealth_shares["Category"].isin(["TopPt1", "RemainingTop1"])]
        .groupby("Date")["Net worth"].sum().reset_index()
        .rename(columns={"Net worth": "Top1Share"}))
bot50 = (wealth_shares[wealth_shares["Category"] == "Bottom50"][["Date", "Net worth"]]
         .rename(columns={"Net worth": "Bot50Share"}).reset_index(drop=True))

# 中国非金融部门杠杆率（BIS，已是 % of GDP，季度，日期为季末）
china_lev = pd.read_csv("中国非金融部门杠杆率.csv")
china_lev["Date"] = pd.to_datetime(china_lev["Date"])
china_lev["ChinaLeverage"] = pd.to_numeric(china_lev["ChinaLeverage"], errors="coerce")
# 季末日期 → 季初日期（与其他数据对齐）：取当季第一天
china_lev["Date"] = china_lev["Date"].dt.to_period("Q").dt.to_timestamp()

# 美国GDP服务业比重（世界银行，年度，1997-2021）
services = pd.read_csv("Services, value added (% of GDP) - US.csv")
services["Date"] = pd.to_datetime(services["Date"])
services["ServicesGDP"] = pd.to_numeric(services["ServicesGDP"], errors="coerce")

# 雇员报酬占比（年度，% of GDI）
emp_comp = pd.read_csv("国内总收入份额：雇员报酬+福利（已付）.csv")
emp_comp = emp_comp.rename(columns={"observation_date": "Date",
                                     emp_comp.columns[1]: "EmpComp"})
emp_comp["Date"] = pd.to_datetime(emp_comp["Date"])
emp_comp["EmpComp"] = pd.to_numeric(emp_comp["EmpComp"], errors="coerce")

# 企业利润占比（年度，% of GDI）
corp_profit = pd.read_csv("国内总收入份额：经调整后的税后企业利润.csv")
corp_profit = corp_profit.rename(columns={"observation_date": "Date",
                                           corp_profit.columns[1]: "CorpProfit"})
corp_profit["Date"] = pd.to_datetime(corp_profit["Date"])
corp_profit["CorpProfit"] = pd.to_numeric(corp_profit["CorpProfit"], errors="coerce")

# 美国国债总额（从1993年，转为万亿美元）
debt = pd.read_csv("美国国债总额_19930401_20260326.csv",
                   usecols=["Record Date", "Total Public Debt Outstanding"])
debt = debt.rename(columns={"Record Date": "Date", "Total Public Debt Outstanding": "Debt"})
debt["Date"] = pd.to_datetime(debt["Date"])
debt["Debt"] = debt["Debt"].astype(float) / 1e12  # 万亿美元

# DGS5 实际利率
dgs5 = pd.read_csv("DGS5.csv").rename(columns={"observation_date": "Date", "DGS5": "DGS5"})
dgs5["Date"] = pd.to_datetime(dgs5["Date"])
dgs5 = dgs5.sort_values("Date").reset_index(drop=True)

cpi = pd.read_csv("CPIAUCSL.csv").rename(columns={"observation_date": "Date", "CPIAUCSL": "CPI"})
cpi["Date"] = pd.to_datetime(cpi["Date"])
cpi = cpi.sort_values("Date").reset_index(drop=True)

# ── 2. 季度化 ─────────────────────────────────────────────────────────────────
gdp_q   = gdp.set_index("Date")["GDP"].resample("QS").last().reset_index()
corp_q  = corp.set_index("Date")["CorpDebt"].resample("QS").last().reset_index()
debt_q  = debt.set_index("Date")["Debt"].resample("QS").last().reset_index()
dgs5_q     = dgs5.set_index("Date")["DGS5"].resample("QS").mean().reset_index()
cpi_q      = cpi.set_index("Date")["CPI"].resample("QS").last().reset_index()
emp_comp_y   = emp_comp.set_index("Date")["EmpComp"].resample("YS").last().reset_index()
corp_profit_y = corp_profit.set_index("Date")["CorpProfit"].resample("YS").last().reset_index()

# ── 3. 计算各指标 ─────────────────────────────────────────────────────────────
# 美国企业杠杆率：CorpDebt(百万) / GDP(十亿) × 0.1 = %
leverage_q = corp_q.merge(gdp_q, on="Date", how="inner").dropna(subset=["CorpDebt", "GDP"])
leverage_q["Leverage"] = leverage_q["CorpDebt"] / leverage_q["GDP"] * 0.1

# 美国国债/GDP：Debt(万亿) × 1000 / GDP(十亿) × 100 = %
debt_gdp = debt_q.merge(gdp_q, on="Date", how="left").dropna(subset=["Debt", "GDP"])
debt_gdp["DebtGDP"] = debt_gdp["Debt"] * 1000 / debt_gdp["GDP"] * 100

# DGS5 实际利率：DGS5 − CPI同比
dgs5_cpi = dgs5_q.merge(cpi_q, on="Date", how="inner")
dgs5_cpi["CPI_YoY"]   = dgs5_cpi["CPI"].pct_change(4) * 100
dgs5_cpi["RealRate"]  = dgs5_cpi["DGS5"] - dgs5_cpi["CPI_YoY"]

# ── 4. 合并所有数据 ───────────────────────────────────────────────────────────
df = leverage_q.merge(china_lev[["Date", "ChinaLeverage"]], on="Date", how="left")
df = df.merge(debt_gdp[["Date", "DebtGDP"]], on="Date", how="left")
df = df.merge(dgs5_cpi[["Date", "RealRate"]], on="Date", how="left")
df = df.merge(emp_comp_y[["Date", "EmpComp"]], on="Date", how="left")
df = df.merge(corp_profit_y[["Date", "CorpProfit"]], on="Date", how="left")
df = df.sort_values("Date").reset_index(drop=True)

# 截取 1975 年之后
df_lev      = df[df["Date"] >= "1975-01-01"].dropna(subset=["Leverage"])
df_china    = df[df["Date"] >= "1975-01-01"].dropna(subset=["ChinaLeverage"])
df_top1     = top1[top1["Date"] >= "1975-01-01"].dropna(subset=["Top1Share"])
df_bot50    = bot50[bot50["Date"] >= "1975-01-01"].dropna(subset=["Bot50Share"])
df_services = services[services["Date"] >= "1975-01-01"].dropna(subset=["ServicesGDP"])
df_debt     = df[df["Date"] >= "1975-01-01"].dropna(subset=["DebtGDP"])
df_rate     = df[df["Date"] >= "1975-01-01"].dropna(subset=["RealRate"])
df_emp      = emp_comp_y[emp_comp_y["Date"] >= "1975-01-01"].dropna(subset=["EmpComp"])
df_profit   = corp_profit_y[corp_profit_y["Date"] >= "1975-01-01"].dropna(subset=["CorpProfit"])

# ── 5. 绘图 ───────────────────────────────────────────────────────────────────
COLOR_US_LEV   = "#1565C0"   # 深蓝：美国企业杠杆率
COLOR_CN_LEV   = "#E65100"   # 橙色：中国非金融部门杠杆率
COLOR_TOP1     = "#B71C1C"   # 深红：前1%财富份额
COLOR_BOT50    = "#00695C"   # 深青：底部50%财富份额
COLOR_SRV      = "#2E7D32"   # 深绿：服务业占GDP比重
COLOR_DEBT     = "#6A1B9A"   # 紫色：美国国债/GDP
COLOR_RATE     = "#00838F"   # 青色：DGS5实际利率
COLOR_EMP      = "#1976D2"   # 蓝色：雇员报酬占比
COLOR_PROFIT   = "#E53935"   # 红色：企业利润占比

start = df_lev["Date"].iloc[0]
end   = df_lev["Date"].iloc[-1]
num_years = (end - start).days / 365.25
fig_w = max(20, num_years * 0.55)

from matplotlib.gridspec import GridSpec

# 上图用 broken axis：上段(emp comp) + 下段(corp profit)，中间断开
# 整体布局：上段(2) + 下段(1.3) + 底图(3.5)
fig = plt.figure(figsize=(fig_w, 20))
gs = GridSpec(3, 1, figure=fig,
              height_ratios=[2, 2, 3.5],
              hspace=0.0)
ax_hi  = fig.add_subplot(gs[0])                   # 上段：emp comp 41-57%
ax_lo  = fig.add_subplot(gs[1], sharex=ax_hi)     # 下段：corp profit -2-11%
ax_bot = fig.add_subplot(gs[2], sharex=ax_hi)     # 底图：信用与债务

# broken axis 外观：隐藏内侧边框，上段无底部、下段无顶部
ax_hi.spines["bottom"].set_visible(False)
ax_lo.spines["top"].set_visible(False)
ax_hi.tick_params(bottom=False, labelbottom=False)
# 上下两段之间留 0.04 的小间隙用于断轴标记
gs.update(hspace=0.04)

# 断轴对角线标记（在两段连接处画斜线）
_d = 0.013
_kw = dict(color="k", clip_on=False, linewidth=1.2, transform=ax_hi.transAxes)
ax_hi.plot((-_d, +_d), (-_d*1.5, +_d*1.5), **_kw)   # 左侧断轴标记（上段底部）
_kw.update(transform=ax_lo.transAxes)
ax_lo.plot((-_d, +_d), (1-_d*1.5, 1+_d*1.5), **_kw)  # 左侧断轴标记（下段顶部）

# 便于后面复用
ax_top = ax_hi   # 标题、legend 放在 ax_hi
ax_lo_panel = ax_lo

# ── 重要事件竖线辅助函数 ──────────────────────────────────────────────────────
events = {
    "1980 沃尔克加息": "1980-01-01",
    "2000 互联网泡沫": "2000-01-01",
    "2008 金融危机":   "2008-09-15",
    "2020 新冠":       "2020-01-01",
}

def add_events(ax, y_text):
    for label, date in events.items():
        ts = pd.Timestamp(date)
        if start <= ts <= end:
            ax.axvline(ts, color="gray", linewidth=1.2, linestyle=":", alpha=0.7, zorder=3)
            ax.text(ts, y_text, label,
                    rotation=90, fontsize=9, color="gray", va="top", ha="right", alpha=0.8)

# ════════════════════════════════════════════════════════════════════════════
# 上图：劳动市场（broken y-axis：上段 emp comp，下段 corp profit，共享左轴）
# ════════════════════════════════════════════════════════════════════════════

# ── 上段（ax_hi）：雇员报酬占比（仅左轴）────────────────────────────────
ec_lo = df_emp["EmpComp"].min() - 1
ec_hi = df_emp["EmpComp"].max() + 1
ax_hi.plot(df_emp["Date"], df_emp["EmpComp"],
           color=COLOR_EMP, linewidth=2.0, label="雇员报酬占比 (% of GDI)", zorder=4)
ax_hi.set_ylim(ec_lo, ec_hi)
ax_hi.set_ylabel("% of GDI", color="black", fontsize=17, labelpad=10)
ax_hi.tick_params(axis="y", labelsize=13)
ax_hi.yaxis.tick_left()

# ── 下段（ax_lo）：企业利润占比（仅左轴）────────────────────────────────
cp_lo = df_profit["CorpProfit"].min() - 1
cp_hi = df_profit["CorpProfit"].max() + 1
ax_lo.plot(df_profit["Date"], df_profit["CorpProfit"],
           color=COLOR_PROFIT, linewidth=2.0, linestyle="--",
           label="企业税后利润占比 (% of GDI)", zorder=4)
ax_lo.set_ylim(cp_lo, cp_hi)
ax_lo.set_ylabel("% of GDI", color="black", fontsize=17, labelpad=10)
ax_lo.tick_params(axis="y", labelsize=13)
ax_lo.yaxis.tick_left()
ax_lo.axhline(0, color=COLOR_PROFIT, linewidth=0.6, linestyle=":", alpha=0.5)

# ── 标题、网格、事件线 ────────────────────────────────────────────────────
ax_hi.set_title("劳动市场：雇员报酬占比 / 企业利润占比 / 财富份额（前1% vs 底部50%）/ 服务业占比",
                fontsize=20, pad=10)
for _ax in [ax_hi, ax_lo]:
    _ax.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    _ax.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
    _ax.set_xlim(start, end)
add_events(ax_hi, ec_hi * 0.98)
add_events(ax_lo, cp_hi * 0.95)

# ── 右轴覆盖层：跨越上下两段的全部高度，绑定 Gini + 服务业 ──────────────
# 必须在 tight_layout 之后获取准确位置
plt.tight_layout()
fig.canvas.draw()

pos_hi = ax_hi.get_position()   # Bbox in figure coords
pos_lo = ax_lo.get_position()
# 合并区域：左=pos_lo.x0, 底=pos_lo.y0, 宽=pos_lo.width, 高=pos_hi.y1 - pos_lo.y0
ax_ov = fig.add_axes(
    [pos_lo.x0, pos_lo.y0, pos_lo.width, pos_hi.y1 - pos_lo.y0],
    facecolor="none", label="top_overlay"
)
# 透明，不干扰左轴
ax_ov.patch.set_alpha(0)
ax_ov.xaxis.set_visible(False)
ax_ov.yaxis.set_visible(False)
for sp in ax_ov.spines.values():
    sp.set_visible(False)
ax_ov.set_xlim(ax_hi.get_xlim())   # 与 ax_hi 共享 x 范围

# 前1% 和底部50% 财富份额（共享右轴，同一 scale）
ax_t_g = ax_ov.twinx()
w_lo = df_bot50["Bot50Share"].min() - 2
w_hi = df_top1["Top1Share"].max() + 2
ax_t_g.plot(df_top1["Date"], df_top1["Top1Share"],
            color=COLOR_TOP1, linewidth=2.0, linestyle="-",
            label="前1%净资产份额 (%)", zorder=4)
ax_t_g.fill_between(df_top1["Date"], df_top1["Top1Share"], 0,
                    color=COLOR_TOP1, alpha=0.12, interpolate=True)
ax_t_g.plot(df_bot50["Date"], df_bot50["Bot50Share"],
            color=COLOR_BOT50, linewidth=2.0, linestyle="--",
            label="底部50%净资产份额 (%)", zorder=4)
ax_t_g.fill_between(df_bot50["Date"], df_bot50["Bot50Share"], 0,
                    color=COLOR_BOT50, alpha=0.15, interpolate=True)
ax_t_g.set_ylabel("财富份额 (%)", color="black", fontsize=17)
ax_t_g.tick_params(axis="y", labelsize=13)
ax_t_g.set_ylim(w_lo, w_hi)

# 服务业/GDP（覆盖层右轴偏移 80）
ax_t_s = ax_ov.twinx()
ax_t_s.spines["right"].set_position(("outward", 80))
ax_t_s.plot(df_services["Date"], df_services["ServicesGDP"],
            color=COLOR_SRV, linewidth=2.0, linestyle=(0, (3, 1, 1, 1)),
            label="美国服务业/GDP (%)", zorder=4, marker="s", markersize=4)
ax_t_s.set_ylim(df_services["ServicesGDP"].min() - 3, df_services["ServicesGDP"].max() + 3)
ax_t_s.set_ylabel("服务业/GDP (%)", color=COLOR_SRV, fontsize=17)
ax_t_s.tick_params(axis="y", labelcolor=COLOR_SRV, labelsize=13)

# ── 合并 legend（放在上段）────────────────────────────────────────────────
h_t, l_t = [], []
for ax in [ax_hi, ax_lo, ax_t_g, ax_t_s, ax_ov]:
    h, l = ax.get_legend_handles_labels()
    h_t += [x for x, lb in zip(h, l) if not lb.startswith("_")]
    l_t += [lb for lb in l if not lb.startswith("_")]
ax_hi.legend(h_t, l_t, loc="upper left", fontsize=15, framealpha=0.85)

# ════════════════════════════════════════════════════════════════════════════
# 下图：信用与债务（美国企业杠杆率 + 中国杠杆率 + 美国国债/GDP + DGS5实际利率）
# ════════════════════════════════════════════════════════════════════════════
ax_b1 = ax_bot
ax_b2 = ax_bot.twinx()   # 美国国债/GDP
ax_b3 = ax_bot.twinx()   # DGS5实际利率

# 美国企业杠杆率（左轴）
ax_b1.plot(df_lev["Date"], df_lev["Leverage"],
           color=COLOR_US_LEV, linewidth=2.0, label="美国非金融企业负债/GDP (%)", zorder=4)
ax_b1.fill_between(df_lev["Date"], df_lev["Leverage"], 0,
                   alpha=0.12, color=COLOR_US_LEV, interpolate=True)

# 中国非金融部门杠杆率（左轴）
ax_b1.plot(df_china["Date"], df_china["ChinaLeverage"],
           color=COLOR_CN_LEV, linewidth=2.0, linestyle="--",
           label="中国非金融部门负债/GDP (%)", zorder=4)
ax_b1.fill_between(df_china["Date"], df_china["ChinaLeverage"], 0,
                   alpha=0.10, color=COLOR_CN_LEV, interpolate=True)

combined_max = max(df_lev["Leverage"].max(), df_china["ChinaLeverage"].max())
ax_b1.set_ylim(0, combined_max * 1.15)
ax_b1.set_ylabel("杠杆率 (% of GDP)", color="black", fontsize=17)
ax_b1.tick_params(axis="y", labelsize=13)

# 美国国债/GDP（右轴）
ax_b2.plot(df_debt["Date"], df_debt["DebtGDP"],
           color=COLOR_DEBT, linewidth=2.0, linestyle=(0, (5, 2)),
           label="美国国债/GDP (%)", zorder=4)
ax_b2.fill_between(df_debt["Date"], df_debt["DebtGDP"], 0,
                   alpha=0.10, color=COLOR_DEBT, interpolate=True)
ax_b2.set_ylim(0, df_debt["DebtGDP"].max() * 1.15)
ax_b2.set_ylabel("美国国债/GDP (%)", color=COLOR_DEBT, fontsize=17)
ax_b2.tick_params(axis="y", labelcolor=COLOR_DEBT, labelsize=13)

# DGS5实际利率（右轴偏移 80，0轴填充）
ax_b3.spines["right"].set_position(("outward", 80))
ax_b3.plot(df_rate["Date"], df_rate["RealRate"],
           color=COLOR_RATE, linewidth=2.0, linestyle=(0, (2, 1)),
           label="DGS5实际利率 (%)", zorder=4)
ax_b3.axhline(0, color=COLOR_RATE, linewidth=0.8, linestyle="--", alpha=0.5)
ax_b3.fill_between(df_rate["Date"], df_rate["RealRate"], 0,
                   where=df_rate["RealRate"] >= 0, color=COLOR_RATE, alpha=0.12, interpolate=True)
ax_b3.fill_between(df_rate["Date"], df_rate["RealRate"], 0,
                   where=df_rate["RealRate"] < 0,  color=COLOR_RATE, alpha=0.12, interpolate=True)
rr_lo, rr_hi = df_rate["RealRate"].min(), df_rate["RealRate"].max()
ax_b3.set_ylim(rr_lo - 2, rr_hi + 2)
ax_b3.set_ylabel("DGS5实际利率 (%)", color=COLOR_RATE, fontsize=17)
ax_b3.tick_params(axis="y", labelcolor=COLOR_RATE, labelsize=13)

ax_bot.text(0.5, 0.97, "信用与债务：企业杠杆率 / 中国杠杆率 / 国债/GDP / 实际利率",
            transform=ax_bot.transAxes, fontsize=20,
            ha="center", va="top", zorder=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=4))
ax_bot.set_xlabel("Date", fontsize=18)
ax_bot.xaxis.set_major_locator(mdates.YearLocator(5))
ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_bot.xaxis.set_minor_locator(mdates.YearLocator(1))
ax_bot.tick_params(axis="x", labelsize=15, rotation=45)
ax_bot.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax_bot.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax_bot.set_xlim(start, end)
add_events(ax_b1, combined_max * 1.12)

h_b, l_b = [], []
for ax in [ax_b1, ax_b2, ax_b3]:
    h, l = ax.get_legend_handles_labels()
    h_b += [x for x, lb in zip(h, l) if not lb.startswith("_")]
    l_b += [lb for lb in l if not lb.startswith("_")]
ax_b1.legend(h_b, l_b, loc="upper left", fontsize=15, framealpha=0.85)

fig.suptitle("利率病时期综合分析", fontsize=30, fontweight="bold", y=0.968)

outfile = "利率病时期_杠杆率与基尼系数.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()
print(f"Chart saved → {outfile}")
