import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

plt.rcParams["font.family"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]

# ── 1. 加载数据 ────────────────────────────────────────────────────────────────
BASE = "/Users/laikaiyuan/Downloads/读书/黄金大师课/Gold Price/"

# 美元-人民币汇率 → 人民币-美元 = 1 / EXCHUS
usdcny = pd.read_csv(BASE + "美元-人民币.csv")
usdcny = usdcny.rename(columns={"observation_date": "Date", "EXCHUS": "USDCNY"})
usdcny["Date"] = pd.to_datetime(usdcny["Date"])
usdcny["CNHUSD"] = 1 / usdcny["USDCNY"]   # 人民币-美元汇率

# 美国向中国出口 / 从中国进口 → 贸易逆差（负数=逆差）
exp_cn = pd.read_csv(BASE + "美国向中国出口货物.csv")
exp_cn = exp_cn.rename(columns={"observation_date": "Date", "EXPCH": "ExpCN"})
exp_cn["Date"] = pd.to_datetime(exp_cn["Date"])

imp_cn = pd.read_csv(BASE + "美国从中国进口货物.csv")
imp_cn = imp_cn.rename(columns={"observation_date": "Date", "IMPCH": "ImpCN"})
imp_cn["Date"] = pd.to_datetime(imp_cn["Date"])

trade_cn = exp_cn.merge(imp_cn, on="Date")
trade_cn["TradeBalance"] = trade_cn["ExpCN"] - trade_cn["ImpCN"]   # 负=逆差，百万美元

# 制造业占GDP百分比（年度）
mfg_gdp = pd.read_csv(BASE + "行业增加值：制造业占GDP的百分比.csv")
mfg_gdp = mfg_gdp.rename(columns={"observation_date": "Date", "VAPGDPMA": "MfgGDP"})
mfg_gdp["Date"] = pd.to_datetime(mfg_gdp["Date"])

# 工业生产总指数（月度）
indpro = pd.read_csv(BASE + "工业生产总指数.csv")
indpro = indpro.rename(columns={"observation_date": "Date", "INDPRO": "INDPRO"})
indpro["Date"] = pd.to_datetime(indpro["Date"])

# 制造业就业人数（千人）
manemp = pd.read_csv(BASE + "制造业就业人数.csv")
manemp = manemp.rename(columns={"observation_date": "Date", "MANEMP": "MfgEmp"})
manemp["Date"] = pd.to_datetime(manemp["Date"])

# ── 2. 月度化对齐 ──────────────────────────────────────────────────────────────
usdcny_m  = usdcny.set_index("Date")["CNHUSD"].resample("MS").last().reset_index()
trade_m   = trade_cn.set_index("Date")["TradeBalance"].resample("MS").last().reset_index()
indpro_m  = indpro.set_index("Date")["INDPRO"].resample("MS").last().reset_index()
manemp_m  = manemp.set_index("Date")["MfgEmp"].resample("MS").last().reset_index()
# 制造业GDP为季度/年度，单独保留
mfg_gdp_m = mfg_gdp.set_index("Date")["MfgGDP"].resample("QS").last().reset_index()

# ── 3. 截取 1985 年起 ──────────────────────────────────────────────────────────
START = "1990-01-01"
usdcny_m  = usdcny_m[usdcny_m["Date"] >= START].reset_index(drop=True)
trade_m   = trade_m[trade_m["Date"] >= START].reset_index(drop=True)
indpro_m  = indpro_m[indpro_m["Date"] >= START].reset_index(drop=True)
manemp_m  = manemp_m[manemp_m["Date"] >= START].reset_index(drop=True)
mfg_gdp_m = mfg_gdp_m[mfg_gdp_m["Date"] >= START].reset_index(drop=True)

# ── 4. 颜色 ───────────────────────────────────────────────────────────────────
COLOR_CNY     = "#C62828"   # 深红：人民币汇率
COLOR_TRADE   = "#E65100"   # 橙：贸易逆差
COLOR_MFGGDP  = "#1565C0"   # 深蓝：制造业/GDP
COLOR_INDPRO  = "#2E7D32"   # 深绿：工业生产指数
COLOR_MANEMP  = "#6A1B9A"   # 紫：制造业就业

# ── 5. 动态图宽 ───────────────────────────────────────────────────────────────
start_dt = usdcny_m["Date"].iloc[0]
end_dt   = usdcny_m["Date"].iloc[-1]
num_years = (end_dt - start_dt).days / 365.25
fig_w = max(22, num_years * 0.55)

fig, ax1 = plt.subplots(figsize=(fig_w, 12))
ax2 = ax1.twinx()   # 贸易逆差
ax3 = ax1.twinx()   # 制造业/GDP
ax4 = ax1.twinx()   # 工业生产指数
ax5 = ax1.twinx()   # 制造业就业

def lift(ax, series, lift_factor=1.8, top_pad=0.1):
    """把线推到图表上半部分。"""
    lo, hi = series.min(), series.max()
    rng = hi - lo if hi != lo else (abs(hi) if hi != 0 else 1)
    ax.set_ylim(lo - rng * lift_factor, hi + rng * top_pad)

def center(ax, series, top_pad=0.15):
    """让线条居中显示。"""
    lo, hi = series.min(), series.max()
    rng = hi - lo if hi != lo else (abs(hi) if hi != 0 else 1)
    ax.set_ylim(lo - rng * 0.5, hi + rng * top_pad)

# ── 人民币/美元汇率（左轴，显示刻度）─────────────────────────────────────────
ax1.plot(usdcny_m["Date"], usdcny_m["CNHUSD"],
         color=COLOR_CNY, linewidth=2.0, label="人民币/美元汇率", zorder=5)
center(ax1, usdcny_m["CNHUSD"])
ax1.set_ylabel("人民币/美元汇率", color=COLOR_CNY, fontsize=17)
ax1.tick_params(axis="y", labelcolor=COLOR_CNY, labelsize=13)

# ── 对华贸易逆差（柱状图，ax2）────────────────────────────────────────────────
bar_width = pd.Timedelta(days=20)
ax2.bar(trade_m["Date"], trade_m["TradeBalance"],
        width=bar_width, color=COLOR_TRADE, alpha=0.55,
        label="对华贸易差额（百万美元）", zorder=3)
ax2.axhline(0, color=COLOR_TRADE, linewidth=0.8, linestyle="--", alpha=0.5)
t_lo = trade_m["TradeBalance"].min()
t_hi = trade_m["TradeBalance"].max()
t_rng = t_hi - t_lo
ax2.set_ylim(t_lo - t_rng * 0.15, t_hi + t_rng * 1.8)   # 柱子在下半部分
ax2.set_ylabel("对华贸易差额（百万美元）", color=COLOR_TRADE, fontsize=17)
ax2.tick_params(axis="y", labelcolor=COLOR_TRADE, labelsize=13)

# ── 制造业/GDP（ax3，右轴偏移 80）────────────────────────────────────────────
ax3.spines["right"].set_position(("outward", 80))
ax3.plot(mfg_gdp_m["Date"], mfg_gdp_m["MfgGDP"],
         color=COLOR_MFGGDP, linewidth=2.0, linestyle="--",
         label="制造业增加值/GDP (%)", zorder=4, marker="o", markersize=4)
lift(ax3, mfg_gdp_m["MfgGDP"], lift_factor=1.5)
ax3.set_ylabel("制造业/GDP (%)", color=COLOR_MFGGDP, fontsize=17)
ax3.tick_params(axis="y", labelcolor=COLOR_MFGGDP, labelsize=13)

# ── 工业生产指数（ax4，隐藏 y 轴）────────────────────────────────────────────
ax4.plot(indpro_m["Date"], indpro_m["INDPRO"],
         color=COLOR_INDPRO, linewidth=1.8, linestyle="-.",
         label="工业生产总指数", zorder=4)
lift(ax4, indpro_m["INDPRO"], lift_factor=1.2)
ax4.yaxis.set_visible(False)

# ── 制造业就业（ax5，隐藏 y 轴）──────────────────────────────────────────────
ax5.plot(manemp_m["Date"], manemp_m["MfgEmp"],
         color=COLOR_MANEMP, linewidth=1.8, linestyle=(0, (5, 2)),
         label="制造业就业人数（千人）", zorder=4)
lift(ax5, manemp_m["MfgEmp"], lift_factor=1.0)
ax5.yaxis.set_visible(False)

# ── 重要事件竖线 ──────────────────────────────────────────────────────────────
events = {
    "1994 汇改":           "1994-01-01",
    "2001 加入WTO":        "2001-12-11",
    "2005 汇改浮动":       "2005-07-21",
    "2008 金融危机":       "2008-09-15",
    "2018 贸易战开始":     "2018-03-22",
    "2020 新冠":           "2020-01-01",
}
y_text = usdcny_m["CNHUSD"].max() * 1.02
for label, date in events.items():
    ts = pd.Timestamp(date)
    if start_dt <= ts <= end_dt:
        ax1.axvline(ts, color="gray", linewidth=1.2, linestyle=":", alpha=0.7, zorder=6)
        ax1.text(ts, y_text, label,
                 rotation=90, fontsize=9, color="gray",
                 va="bottom", ha="right", alpha=0.85)

# ── x 轴格式 ──────────────────────────────────────────────────────────────────
ax1.set_xlabel("Date", fontsize=18)
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.tick_params(axis="x", labelsize=15, rotation=45)
ax1.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax1.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
ax1.set_xlim(start_dt, end_dt)

# ── Legend ────────────────────────────────────────────────────────────────────
all_h, all_l = [], []
for ax in [ax1, ax2, ax3, ax4, ax5]:
    h, l = ax.get_legend_handles_labels()
    all_h += [x for x, lb in zip(h, l) if not lb.startswith("_")]
    all_l += [lb for lb in l if not lb.startswith("_")]
ax1.legend(all_h, all_l, loc="upper center", bbox_to_anchor=(0.5, -0.1),
           ncol=3, fontsize=16, framealpha=0.85)

plt.title("中美贸易与美国制造业：汇率 / 对华贸易差额 / 制造业/GDP / 工业生产指数 / 制造业就业",
          fontsize=20, pad=14)
plt.tight_layout()
outfile = BASE + "中美贸易与制造业.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()
print(f"Chart saved → {outfile}")
