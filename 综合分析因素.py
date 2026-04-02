import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

# 支持中文标题
plt.rcParams["font.family"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]

# ── 1. 加载数据 ────────────────────────────────────────────────────────────────
def load_value_csv(path):
    df = pd.read_csv(path)
    df["Date"]  = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Value"] = df["Value"].astype(str).str.replace(",", "").astype(float)
    return df.sort_values("Date").reset_index(drop=True)

dj    = load_value_csv("from 1914 Dow Jones Industrial Average Historical Data.csv")
sp    = load_value_csv("from 1927 S&P Industrial Average Historical Data.csv")
wti   = load_value_csv("from 1947 WTI crude oil prices.csv")

dgs5  = pd.read_csv("DGS5.csv").rename(columns={"observation_date": "Date", "DGS5": "DGS5"})
dgs5["Date"] = pd.to_datetime(dgs5["Date"])
dgs5  = dgs5.sort_values("Date").reset_index(drop=True)

cpi   = pd.read_csv("CPIAUCSL.csv").rename(columns={"observation_date": "Date", "CPIAUCSL": "CPI"})
cpi["Date"] = pd.to_datetime(cpi["Date"])
cpi   = cpi.sort_values("Date").reset_index(drop=True)

gold  = pd.read_csv("monthly-gold-price.csv")
gold["Date"] = pd.to_datetime(gold["Date"])
gold  = gold.sort_values("Date").reset_index(drop=True)

trade = pd.read_csv("Trade Balance Goods and Services.csv")
trade = trade.rename(columns={"observation_date": "Date", "BOPGSTB": "TradeBalance"})
trade["Date"] = pd.to_datetime(trade["Date"])
trade["TradeBalance"] = trade["TradeBalance"].astype(float) / 1000  # 转为十亿美元
trade = trade.sort_values("Date").reset_index(drop=True)

expgs = pd.read_csv("EXPGS.csv").rename(columns={"observation_date": "Date", "EXPGS": "EXPGS"})
expgs["Date"] = pd.to_datetime(expgs["Date"])
impgs = pd.read_csv("IMPGS.csv").rename(columns={"observation_date": "Date", "IMPGS": "IMPGS"})
impgs["Date"] = pd.to_datetime(impgs["Date"])

gdp = pd.read_csv("1947-present GDP.csv")
gdp = gdp.rename(columns={"observation_date": "Date"})
gdp["Date"] = pd.to_datetime(gdp["Date"])
gdp = gdp.sort_values("Date").reset_index(drop=True)

debt  = pd.read_csv("美国国债总额_19930401_20260326.csv",
                    usecols=["Record Date", "Total Public Debt Outstanding"])
debt  = debt.rename(columns={"Record Date": "Date",
                              "Total Public Debt Outstanding": "Debt"})
debt["Date"]  = pd.to_datetime(debt["Date"])
debt["Debt"]  = debt["Debt"].astype(float) / 1e12   # 转换为万亿美元
debt  = debt.sort_values("Date").reset_index(drop=True)

# ── 2. 季度化 ─────────────────────────────────────────────────────────────────
dj_q   = dj.set_index("Date")["Value"].resample("QS").last().reset_index()
sp_q   = sp.set_index("Date")["Value"].resample("QS").last().reset_index()
wti_q  = wti.set_index("Date")["Value"].resample("QS").last().reset_index()
dgs5_q = dgs5.set_index("Date")["DGS5"].resample("QS").mean().reset_index()
cpi_q  = cpi.set_index("Date")["CPI"].resample("QS").last().reset_index()
gold_q = gold.set_index("Date")["Price"].resample("QS").last().reset_index()
gdp_q   = gdp[["Date", "GDP"]].copy()  # GDP 已是季度数据
debt_q  = debt.set_index("Date")["Debt"].resample("QS").last().reset_index()
debt_q  = debt_q.merge(gdp_q, on="Date", how="left")
debt_q["DebtGDP"] = debt_q["Debt"] * 1000 / debt_q["GDP"] * 100   # 美债/GDP (%) — Debt万亿→十亿，GDP十亿
trade_q = trade.set_index("Date")["TradeBalance"].resample("QS").sum().reset_index()
# TradeBalance 已在加载时转为十亿美元，此处季度合计后直接使用

# 合并计算 Trade Balance / GDP（%）：季度合计×4 年化后 ÷ 年化GDP
trade_q = trade_q.merge(gdp_q, on="Date", how="left")
trade_q["TradeGDP"] = trade_q["TradeBalance"] * 4 / trade_q["GDP"] * 100
# 忽略 2026 Q1
trade_q = trade_q[trade_q["Date"] < "2026-01-01"]

# EXPGS - IMPGS 贸易差额 / GDP（%）
expgs_q = expgs.set_index("Date")["EXPGS"].resample("QS").last().reset_index()
impgs_q = impgs.set_index("Date")["IMPGS"].resample("QS").last().reset_index()
trade_expgs_q = expgs_q.merge(impgs_q, on="Date")
trade_expgs_q["TradeBalance_EXPGS"] = trade_expgs_q["EXPGS"] - trade_expgs_q["IMPGS"]
trade_expgs_q = trade_expgs_q.merge(gdp_q, on="Date", how="left")
trade_expgs_q["TradeGDP_EXPGS"] = trade_expgs_q["TradeBalance_EXPGS"] / trade_expgs_q["GDP"] * 100

# ── 3. 计算导数 ───────────────────────────────────────────────────────────────
dj_q["dDJ"]   = dj_q["Value"].pct_change() * 100
sp_q["dSP"]   = sp_q["Value"].pct_change() * 100
wti_q["dWTI"] = wti_q["Value"].pct_change() * 100

# DGS5 实际利率 = DGS5 − CPI同比，再取季度变化率
dgs5_cpi = dgs5_q.merge(cpi_q, on="Date")
dgs5_cpi["CPI_YoY"]     = dgs5_cpi["CPI"].pct_change(4) * 100
dgs5_cpi["Real_DGS5"]   = dgs5_cpi["DGS5"] - dgs5_cpi["CPI_YoY"]
dgs5_cpi["dReal_DGS5"]  = dgs5_cpi["Real_DGS5"].diff()  # 利率已是%，用差值(%pt)而非变化率

# ── 4. Merge ──────────────────────────────────────────────────────────────────
df = (dj_q[["Date", "Value"]].rename(columns={"Value": "DJ"})
      .merge(sp_q[["Date", "Value"]].rename(columns={"Value": "SP"}),  on="Date")
      .merge(wti_q[["Date", "Value"]].rename(columns={"Value": "WTI"}), on="Date")
      .merge(dgs5_cpi[["Date", "Real_DGS5"]],   on="Date")
      .merge(gold_q[["Date", "Price"]],           on="Date")
      .merge(debt_q[["Date", "Debt", "DebtGDP"]],   on="Date", how="left")
      .merge(trade_q[["Date", "TradeGDP"]],                    on="Date", how="left")
      .merge(trade_expgs_q[["Date", "TradeGDP_EXPGS", "TradeBalance_EXPGS"]], on="Date", how="left")
      .dropna(subset=["DJ", "SP", "WTI", "Real_DGS5", "Price"])
      .sort_values("Date")
      .reset_index(drop=True))

# ── 5. 绘图函数 ───────────────────────────────────────────────────────────────
COLOR_DJ   = "#E65100"
COLOR_SP   = "#7B1FA2"
COLOR_WTI  = "#5D4037"
COLOR_DGS5 = "#1565C0"
COLOR_GOLD = "#D4A017"

COLOR_TRADE = "#00897B"   # 青绿色

def make_chart(data, start_year, end_year, outfile, use_debt=False, show_wti=True, show_trade=False, trade_col="TradeGDP"):
    d = data[(data["Date"].dt.year >= start_year) &
             (data["Date"].dt.year <= end_year)].reset_index(drop=True)

    num_years = (d["Date"].iloc[-1] - d["Date"].iloc[0]).days / 365.25
    fig, ax1 = plt.subplots(figsize=(max(20, num_years * 0.55), 16))

    # 5条线各自独立轴
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    ax5 = ax1.twinx()
    ax6 = ax1.twinx()

    # 柱状图宽度（季度 ≈ 60天）
    bar_width = pd.Timedelta(days=55)

    if use_debt:
        # 国债/GDP 替换 DJ 和 S&P（合并为单一柱状图）
        ax1.bar(d["Date"], d["DebtGDP"], width=bar_width, color="#B71C1C", alpha=0.6,
                label="US Debt / GDP (%)", zorder=3)
        ax1.set_ylim(0, d["DebtGDP"].max() * 2.5)
        ax1.yaxis.set_visible(False)
        ax2.yaxis.set_visible(False)   # ax2 不绘制，只隐藏
    else:
        # DJ 柱状图
        ax1.bar(d["Date"], d["DJ"], width=bar_width, color=COLOR_DJ, alpha=0.6,
                label="Dow Jones", zorder=3)
        ax1.set_ylim(0, d["DJ"].max() * 2.5)
        ax1.yaxis.set_visible(False)

        # S&P 柱状图
        ax2.bar(d["Date"], d["SP"], width=bar_width, color=COLOR_SP, alpha=0.5,
                label="S&P 500", zorder=2)
        ax2.set_ylim(0, d["SP"].max() * 2.5)
        ax2.yaxis.set_visible(False)

    line_series = [
        (ax4, "Real_DGS5", COLOR_DGS5, "Real Rate DGS5 (%)",   1.2),
        (ax5, "Price",     COLOR_GOLD, "Gold Price (USD)",      1.5),
    ]
    if show_wti:
        line_series = [(ax3, "WTI", COLOR_WTI, "WTI Crude Oil (USD)", 1.2)] + line_series
    else:
        ax3.yaxis.set_visible(False)

    def set_line_ylim(ax, series, lift=1.8):
        """把线条推到图表上半部分：在数据下方留出 lift 倍数据范围的空白。"""
        lo = series.min()
        hi = series.max()
        rng = hi - lo if hi != lo else (abs(hi) if hi != 0 else 1)
        ax.set_ylim(lo - rng * lift, hi + rng * 0.1)

    for ax, col, color, label, lw in line_series:
        ax.plot(d["Date"], d[col], color=color, linewidth=lw, label=label, zorder=4)
        # 黄金放中间（lift=0.1），其他线条推到上半部分（lift=1.8）
        set_line_ylim(ax, d[col], lift=0.1 if col == "Price" else 1.8)
        ax.yaxis.set_visible(False)
        if col == "Real_DGS5":
            ax.axhline(0, color=COLOR_DGS5, linewidth=0.8, linestyle="--", alpha=0.6, zorder=3)
            ax.fill_between(d["Date"], d[col], 0,
                            where=d[col] >= 0, color=COLOR_DGS5, alpha=0.15, interpolate=True)
            ax.fill_between(d["Date"], d[col], 0,
                            where=d[col] < 0,  color=COLOR_DGS5, alpha=0.15, interpolate=True)

    # Trade Balance（可选）
    trade_labels = {
        "TradeGDP":       "Trade Balance / GDP (%)",
        "TradeGDP_EXPGS": "Trade Balance / GDP (%)",
        "TradeBalance_EXPGS": "Trade Balance (Billion USD)",
    }
    if show_trade and trade_col in d.columns:
        tb = d.dropna(subset=[trade_col])
        ax6.plot(tb["Date"], tb[trade_col], color=COLOR_TRADE, linewidth=1.2,
                 label=trade_labels.get(trade_col, "Trade Balance"), zorder=4)
        set_line_ylim(ax6, tb[trade_col])
        ax6.axhline(0, color=COLOR_TRADE, linewidth=0.6, linestyle="--", alpha=0.5)
        ax6.fill_between(tb["Date"], tb[trade_col], 0,
                         where=tb[trade_col] >= 0, color=COLOR_TRADE, alpha=0.12, interpolate=True)
        ax6.fill_between(tb["Date"], tb[trade_col], 0,
                         where=tb[trade_col] < 0,  color=COLOR_TRADE, alpha=0.12, interpolate=True)
    ax6.yaxis.set_visible(False)

    # 仅在 2000-2026 图：显示左右 y 轴
    if use_debt and show_trade and trade_col in d.columns:
        tb2 = d.dropna(subset=[trade_col])

        # 左轴：贸易差额/GDP（ax6 移到左侧）
        ax6.yaxis.set_visible(True)
        ax6.yaxis.set_label_position("left")
        ax6.yaxis.tick_left()
        t_min, t_max = tb2[trade_col].min(), tb2[trade_col].max()
        ax6.set_yticks(np.linspace(t_min, t_max, 6))
        ax6.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax6.set_ylabel("贸易差额/GDP (%)", color=COLOR_TRADE, fontsize=18)
        ax6.tick_params(axis="y", labelcolor=COLOR_TRADE, labelsize=14)

        # 右轴：美债/GDP（ax1 移到右侧）
        ax1.yaxis.set_visible(True)
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.tick_right()
        d_max = d["DebtGDP"].dropna().max()
        ax1.set_yticks(np.linspace(0, d_max, 6))
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax1.set_ylabel("美债/GDP (%)", color="#B71C1C", fontsize=18)
        ax1.tick_params(axis="y", labelcolor="#B71C1C", labelsize=14)

    elif not use_debt and show_trade and trade_col in d.columns:
        tb2 = d.dropna(subset=[trade_col])

        # 左轴：贸易差额/GDP（ax6 移到左侧）
        ax6.yaxis.set_visible(True)
        ax6.yaxis.set_label_position("left")
        ax6.yaxis.tick_left()
        t_min, t_max = tb2[trade_col].min(), tb2[trade_col].max()
        ax6.set_yticks(np.linspace(t_min, t_max, 6))
        ax6.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax6.set_ylabel("贸易差额/GDP (%)", color=COLOR_TRADE, fontsize=18)
        ax6.tick_params(axis="y", labelcolor=COLOR_TRADE, labelsize=14)

        # 右轴：道琼斯（ax1 移到右侧）
        ax1.yaxis.set_visible(True)
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.tick_right()
        dj_max = d["DJ"].dropna().max()
        ax1.set_yticks(np.linspace(0, dj_max, 6))
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax1.set_ylabel("道琼斯指数", color=COLOR_DJ, fontsize=18)
        ax1.tick_params(axis="y", labelcolor=COLOR_DJ, labelsize=14)

    # 在左侧用文字标注各轴的 y=0 位置
    bar2_label = "债/GDP=0" if use_debt else "S&P=0"
    wti_label  = "原油=0" if show_wti else None
    trade_label_map = {
        "TradeGDP": "贸易/GDP=0", "TradeGDP_EXPGS": "贸易/GDP=0",
        "TradeBalance_EXPGS": "贸易差额=0",
    }
    trade_label = trade_label_map.get(trade_col) if show_trade else None
    all_axes_info = [
        (ax1, COLOR_DJ,                                "道琼斯=0"),
        (ax2, "#B71C1C" if use_debt else COLOR_SP,     bar2_label),
        (ax3, COLOR_WTI,                               wti_label),
        (ax4, COLOR_DGS5,                              "实际利率=0"),
        (ax5, COLOR_GOLD,                              "黄金=0"),
        (ax6, COLOR_TRADE,                             trade_label),
    ]
    for ax, color, label in all_axes_info:
        if label is None:
            continue
        y_min, y_max = ax.get_ylim()
        if y_min < 0 < y_max and (y_max - y_min) > 0:
            y_frac = (0 - y_min) / (y_max - y_min)
            ax.annotate(label, xy=(-0.01, y_frac),
                        xycoords=("axes fraction", "axes fraction"),
                        fontsize=13, color=color, ha="right", va="center",
                        fontweight="bold")

    # x轴格式（只用ax1控制）
    ax1.set_xlabel("Date", fontsize=23)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.tick_params(axis="x", labelsize=20, rotation=45)
    ax1.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
    ax1.set_xlim(d["Date"].iloc[0], d["Date"].iloc[-1])

    # 1980 Q1 竖线
    ax1.axvline(pd.Timestamp("1980-01-01"), color="red", linewidth=1.5,
                linestyle="--", alpha=0.7, zorder=6)

    # 合并legend
    all_h, all_l = [], []
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        h, l = ax.get_legend_handles_labels()
        all_h += [x for x, lb in zip(h, l) if not lb.startswith("_")]
        all_l += [lb for lb in l if not lb.startswith("_")]
    ax1.legend(all_h, all_l, loc="upper center", bbox_to_anchor=(0.5, -0.1),
               ncol=5, fontsize=20, framealpha=0.85)

    if use_debt:
        title = f"综合分析因素：美债/GDP / 贸易差额/GDP / DGS5实际利率 / 黄金价格 ({start_year}–{end_year})"
    else:
        title = f"综合分析因素：道琼斯 / S&P / 原油 / 贸易差额/GDP / DGS5实际利率 / 黄金价格 ({start_year}–{end_year})"
    plt.title(title, fontsize=24, pad=16)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Chart saved → {outfile}")

# ── 6. 生成两张图 ─────────────────────────────────────────────────────────────
make_chart(df, 1962, 2000, "综合分析因素_1962-2000.png", use_debt=False, show_trade=True, trade_col="TradeGDP_EXPGS")
make_chart(df, 2000, 2026, "综合分析因素_2000-2026.png", use_debt=True, show_wti=False, show_trade=True, trade_col="TradeGDP")
