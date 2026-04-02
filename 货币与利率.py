import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# ── 字体 ──────────────────────────────────────────────────────────────────
for fn in ["Hiragino Sans GB", "STHeiti", "PingFang SC", "Arial Unicode MS"]:
    try:
        matplotlib.font_manager.findfont(fn, fallback_to_default=False)
        plt.rcParams["font.family"] = fn
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

BASE = "/Users/laikaiyuan/Downloads/读书/黄金大师课/Gold Price/"

# ── 颜色 ──────────────────────────────────────────────────────────────────
C_JPY  = "#C62828"   # 深红  – 日元兑美元
C_EUR  = "#1565C0"   # 深蓝  – 欧元兑美元
C_ECB  = "#7B1FA2"   # 紫    – 欧洲央行利率
C_JGB  = "#00838F"   # 青蓝  – 日本10年期国债收益率
C_USD  = "#E65100"   # 深橙  – 美元指数
C_DGS5 = "#2E7D32"   # 深绿  – DGS5（保留用于轴色）
C_VOL  = "#0277BD"   # 深蓝  – 黄金波动率
C_REAL = "#AD1457"   # 深玫红 – DGS5实际利率
C_GOLD = "#D4A017"   # 金    – 黄金价格

# ── 读取数据 ──────────────────────────────────────────────────────────────
def load_fred(path, val_col, rename=None):
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "Date"})
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col]).sort_values("Date").reset_index(drop=True)
    if rename:
        df = df.rename(columns={val_col: rename})
        val_col = rename
    return df[["Date", val_col]]

jpy_df  = load_fred(BASE + "美元兑日元即期汇率.csv",         "DEXJPUS",        "JPY_USD")
eur_df  = load_fred(BASE + "美元兑欧元即期汇率.csv",         "DEXUSEU",        "EUR_USD")
ecb_df  = load_fred(BASE + "欧洲央行欧元区存款便利利率.csv", "ECBDFR",         "ECB")
jgb_df  = load_fred(BASE + "日本10年期长期政府债券收益率.csv","IRLTLT01JPM156N","JGB10")
usd_df   = load_fred(BASE + "名义美元指数.csv",               "DTWEXBGS",       "USD_IDX")
dgs10_df = load_fred(BASE + "DGS10.csv",                     "DGS10",          "DGS10")
dgs5_df  = load_fred(BASE + "DGS5.csv",                      "DGS5",           "DGS5")
cpi_df   = load_fred(BASE + "CPIAUCSL.csv",                  "CPIAUCSL",       "CPI")

gold_df = pd.read_csv(BASE + "monthly-gold-price.csv", parse_dates=["Date"])
gold_df["Price"] = pd.to_numeric(gold_df["Price"], errors="coerce")
gold_df = gold_df.dropna().sort_values("Date").reset_index(drop=True)

# ── 季度化 ────────────────────────────────────────────────────────────────
def to_q(df, date_col="Date", how="last"):
    df2 = df.set_index(date_col)
    q = df2.resample("QS").last() if how == "last" else df2.resample("QS").mean()
    q.index.name = "Date"
    return q.reset_index()

jpy_q  = to_q(jpy_df,  how="last")
eur_q  = to_q(eur_df,  how="last")
ecb_q  = to_q(ecb_df,  how="last")
jgb_q  = to_q(jgb_df,  how="mean")
usd_q   = to_q(usd_df,   how="last")
dgs10_q = to_q(dgs10_df, how="mean")
dgs5_q  = to_q(dgs5_df,  how="mean")
cpi_q   = to_q(cpi_df,   how="last")
gold_q  = to_q(gold_df[["Date","Price"]], how="last")

# 黄金ETF波动率（CBOE GVZ）
gvz_df  = load_fred(BASE + "CBOE黄金ETF波动率指数.csv",       "GVZCLS", "GoldVol")
gvz_q   = to_q(gvz_df, how="mean")

# 纳斯达克/道琼斯波动率 + 利差
vxn_df   = load_fred(BASE + "CBOE纳斯达克100波动率指数.csv",    "VXNCLS", "VXN")
vxd_df   = load_fred(BASE + "CBOE道琼斯工业平均波动率指数.csv", "VXDCLS", "VXD")
spr_df   = load_fred(BASE + "10年期固定到期国债减去3个月固定到期国债.csv", "T10Y3M", "Spread")
vxn_q    = to_q(vxn_df,  how="mean")
vxd_q    = to_q(vxd_df,  how="mean")
spr_q    = to_q(spr_df,  how="mean")

# DGS5 实际利率 = DGS5 - CPI同比
cpi_q["CPI_YOY"] = cpi_q["CPI"].pct_change(4) * 100
real_q = dgs5_q.merge(cpi_q[["Date","CPI_YOY"]], on="Date", how="inner")
real_q["REAL5"] = real_q["DGS5"] - real_q["CPI_YOY"]
real_q = real_q[["Date","DGS5","REAL5"]]

# ── 合并上图数据（以ECB为基础，1999年起） ────────────────────────────────
top = ecb_q.copy()
for other, col in [(jpy_q,"JPY_USD"), (eur_q,"EUR_USD"), (jgb_q,"JGB10"), (dgs10_q,"DGS10")]:
    top = top.merge(other[["Date", col]], on="Date", how="left")
top = top.sort_values("Date").dropna(subset=["ECB","JPY_USD","EUR_USD","JGB10","DGS10"])
top = top[(top["Date"] >= "2008-01-01") & (top["Date"] <= "2020-10-01")].reset_index(drop=True)

# ── 合并下图数据（以黄金为基础，美元指数2006前留空） ──────────────────────
bot = gold_q.copy()
bot = bot.merge(usd_q[["Date","USD_IDX"]], on="Date", how="left")
bot = bot.merge(real_q[["Date","REAL5"]], on="Date", how="left")
bot = bot.merge(gvz_q[["Date","GoldVol"]], on="Date", how="left")
bot = bot.sort_values("Date").dropna(subset=["Price"])
bot = bot[(bot["Date"] >= "2008-01-01") & (bot["Date"] <= "2020-10-01")].reset_index(drop=True)

# ── 合并第三图数据
bot2 = vxn_q.copy()
bot2 = bot2.merge(vxd_q[["Date","VXD"]],    on="Date", how="left")
bot2 = bot2.merge(spr_q[["Date","Spread"]], on="Date", how="left")
bot2 = bot2.sort_values("Date").dropna(subset=["VXN"])
bot2 = bot2[(bot2["Date"] >= "2008-01-01") & (bot2["Date"] <= "2020-10-01")].reset_index(drop=True)

# 统一时间轴：两图共同起止时间
start = max(top["Date"].iloc[0], bot["Date"].iloc[0])
end   = min(top["Date"].iloc[-1], bot["Date"].iloc[-1])
top = top[(top["Date"] >= start) & (top["Date"] <= end)].reset_index(drop=True)
bot = bot[(bot["Date"] >= start) & (bot["Date"] <= end)].reset_index(drop=True)

print(f"上图范围：{top['Date'].iloc[0].date()} → {top['Date'].iloc[-1].date()}，{len(top)} 季度")
print(f"中图范围：{bot['Date'].iloc[0].date()} → {bot['Date'].iloc[-1].date()}，{len(bot)} 季度")
print(f"下图范围：{bot2['Date'].iloc[0].date()} → {bot2['Date'].iloc[-1].date()}，{len(bot2)} 季度")

# ── 计算图宽 ──────────────────────────────────────────────────────────────
num_years = (top["Date"].iloc[-1] - top["Date"].iloc[0]).days / 365.25
width = max(20, num_years * 0.75)

# ── 绘图布局 ──────────────────────────────────────────────────────────────
fig, (ax_t1, ax_b1, ax_c1) = plt.subplots(3, 1, figsize=(width, 26),
                                    gridspec_kw={"hspace": 0.08, "height_ratios": [3, 2, 2]})
fig.patch.set_facecolor("#FAFAF8")
for ax in [ax_t1, ax_b1, ax_c1]:
    ax.set_facecolor("#FAFAF8")

# ════════════════════════════════════════════════════════
# 上图额外轴
# ════════════════════════════════════════════════════════
# ax_t2：欧元兑美元（右轴 1.0）
ax_t2 = ax_t1.twinx()
ax_t2.spines["right"].set_position(("axes", 1.0))

# ax_t3：利率轴 ECB + JGB（右轴 1.08）
ax_t3 = ax_t1.twinx()
ax_t3.spines["right"].set_position(("axes", 1.08))

for ax in [ax_t2, ax_t3]:
    ax.spines["right"].set_visible(True)

# ════════════════════════════════════════════════════════
# 下图额外轴
# ════════════════════════════════════════════════════════
# ax_b2：实际利率（右轴 1.0）
ax_b2 = ax_b1.twinx()
ax_b2.spines["right"].set_position(("axes", 1.0))
ax_b2.spines["right"].set_visible(True)

# ax_b3：黄金（右轴 1.09）
ax_b3 = ax_b1.twinx()
ax_b3.spines["right"].set_position(("axes", 1.09))
ax_b3.spines["right"].set_visible(True)

# ax_b4：黄金波动率（右轴 1.18）
ax_b4 = ax_b1.twinx()
ax_b4.spines["right"].set_position(("axes", 1.18))
ax_b4.spines["right"].set_visible(True)

# ════════════════════════════════════════════════════════
# 第三图额外轴
# ════════════════════════════════════════════════════════
# ax_c2：利差（右轴 1.0）
ax_c2 = ax_c1.twinx()
ax_c2.spines["right"].set_position(("axes", 1.0))
ax_c2.spines["right"].set_visible(True)

# ════════════════════════════════════════════════════════
# 上图绘线
# ════════════════════════════════════════════════════════
lw = 1.8
t_dates = top["Date"]

ax_t1.plot(t_dates, top["JPY_USD"], color=C_JPY, lw=lw, label="美元兑日元 (USD/JPY)")
ax_t2.plot(t_dates, top["EUR_USD"], color=C_EUR, lw=lw, label="欧元兑美元 (EUR/USD)")
ax_t3.plot(t_dates, top["ECB"],    color=C_ECB,  lw=lw, label="欧央行存款便利利率 (%)")
ax_t3.plot(t_dates, top["JGB10"], color=C_JGB,  lw=lw, label="日本10年期国债收益率 (%)")
ax_t3.plot(t_dates, top["DGS10"], color=C_DGS5, lw=lw, label="美国10年期国债收益率 (%)")

# ════════════════════════════════════════════════════════
# 下图绘线
# ════════════════════════════════════════════════════════
b_dates = bot["Date"]

ax_b1.plot(b_dates, bot["USD_IDX"], color=C_USD,  lw=lw, label="美元指数 (DXY)")

# 实际利率（ax_b2）
ax_b2.plot(b_dates, bot["REAL5"],  color=C_REAL, lw=lw, label="DGS5 实际利率 (%)")
ax_b2.fill_between(b_dates, bot["REAL5"], 0,
                   where=(bot["REAL5"] >= 0), color=C_REAL, alpha=0.18,
                   linewidth=0, interpolate=True)
ax_b2.fill_between(b_dates, bot["REAL5"], 0,
                   where=(bot["REAL5"] <  0), color=C_REAL, alpha=0.30,
                   linewidth=0, interpolate=True)
ax_b2.axhline(0, color="#888888", lw=1.0, ls=":", alpha=0.7, zorder=5)
ax_b2.annotate("0%", xy=(b_dates.iloc[-1], 0),
               xytext=(6, 0), textcoords="offset points",
               fontsize=13, color="#666666", va="center")

# 黄金价格（ax_b3）
ax_b3.plot(b_dates, bot["Price"],  color=C_GOLD, lw=lw+0.4, label="黄金价格 (USD/oz)")
ax_b3.fill_between(b_dates, bot["Price"], bot["Price"].min() * 0.85,
                   color=C_GOLD, alpha=0.13, linewidth=0)

# 黄金波动率（ax_b4）
ax_b4.plot(b_dates, bot["GoldVol"], color=C_VOL, lw=lw, ls="--", label="黄金ETF波动率 GVZ (%)")
ax_b4.fill_between(b_dates, bot["GoldVol"], 0,
                   color=C_VOL, alpha=0.12, linewidth=0, interpolate=True)

# ════════════════════════════════════════════════════════
# 第三图绘线
# ════════════════════════════════════════════════════════
C_VXN  = "#6A1B9A"   # 深紫  – 纳斯达克波动率
C_VXD  = "#00695C"   # 深绿  – 道琼斯波动率
C_SPR  = "#BF360C"   # 深橙红 – 利差

c_dates = bot2["Date"]
ax_c1.plot(c_dates, bot2["VXN"], color=C_VXN, lw=lw,      label="纳斯达克100波动率 VXN (%)")
ax_c1.plot(c_dates, bot2["VXD"], color=C_VXD, lw=lw,      label="道琼斯波动率 VXD (%)")
ax_c1.fill_between(c_dates, bot2["VXN"], 0, color=C_VXN, alpha=0.10, linewidth=0, interpolate=True)
ax_c1.fill_between(c_dates, bot2["VXD"], 0, color=C_VXD, alpha=0.10, linewidth=0, interpolate=True)

ax_c2.plot(c_dates, bot2["Spread"], color=C_SPR, lw=lw, ls="--", label="10Y-3M利差 (%)")
ax_c2.fill_between(c_dates, bot2["Spread"], 0,
                   where=(bot2["Spread"] >= 0), color=C_SPR, alpha=0.15, linewidth=0, interpolate=True)
ax_c2.fill_between(c_dates, bot2["Spread"], 0,
                   where=(bot2["Spread"] <  0), color=C_SPR, alpha=0.35, linewidth=0, interpolate=True)
ax_c2.axhline(0, color="#888888", lw=1.0, ls=":", alpha=0.7, zorder=5)
ax_c2.annotate("0%", xy=(c_dates.iloc[-1], 0),
               xytext=(6, 0), textcoords="offset points",
               fontsize=13, color="#666666", va="center")

# ════════════════════════════════════════════════════════
# 事件标注（上下图同步标注）
# ════════════════════════════════════════════════════════
import matplotlib.patches as mpatches

events = [
    # (日期,        标签,              颜色,     y位置偏移方向)
    ("2012-12-26", "安倍上台",        "#B71C1C", "top"),    # 深红
    ("2013-04-04", "QQE启动",         "#E65100", "bottom"), # 深橙
    ("2014-10-31", "QQE扩大",         "#E65100", "top"),    # 深橙
    ("2016-01-29", "BOJ负利率",       "#E65100", "bottom"), # 深橙
    ("2016-09-21", "YCC",             "#E65100", "top"),    # 深橙
    ("2015-12-16", "Fed首次加息",     "#1A237E", "top"),    # 深蓝
    ("2018-12-20", "Fed末次加息",     "#1A237E", "bottom"), # 深蓝
    ("2019-07-31", "Fed首次降息",     "#00695C", "top"),    # 深绿
]
# 加息周期阴影：2015-12 ~ 2018-12
hike_start = pd.Timestamp("2015-12-01")
hike_end   = pd.Timestamp("2018-12-31")
# 降息周期阴影：2019-07 ~ 2020-12
cut_start  = pd.Timestamp("2019-07-01")
cut_end    = pd.Timestamp("2020-12-31")
# QQE阴影：2013-04 ~ 2020-12（日本QQE持续至今）
qqe_start  = pd.Timestamp("2013-04-04")
qqe_end    = pd.Timestamp("2020-12-31")

# 先画阴影（不依赖ylim）
for ax in [ax_t1, ax_b1, ax_c1]:
    ax.axvspan(qqe_start,  qqe_end,  color="#E65100", alpha=0.05, zorder=0)
    ax.axvspan(hike_start, hike_end, color="#1A237E", alpha=0.07, zorder=0)
    ax.axvspan(cut_start,  cut_end,  color="#00695C", alpha=0.07, zorder=0)
    for date_str, label, color, pos in events:
        ax.axvline(pd.Timestamp(date_str), color=color, lw=1.2, ls="--",
                   alpha=0.75, zorder=6)

# ════════════════════════════════════════════════════════
# 上图 Y轴范围
# 布局：汇率区 [0.55, 0.97]（JPY顶 EUR次）
#       利率区 [0.02, 0.43]（ECB/JGB/DGS10）
# 中间留 ~10% 空白分隔
# ════════════════════════════════════════════════════════
def zone_ylim(d_min, d_max, za, zb):
    """将数据推入 [za, zb] 区域 (0=底, 1=顶)"""
    d_rng = d_max - d_min or 1e-6
    return d_min - za / (zb - za) * d_rng, d_max + (1 - zb) / (zb - za) * d_rng

j_min, j_max = top["JPY_USD"].min(), top["JPY_USD"].max()
e_min, e_max = top["EUR_USD"].min(), top["EUR_USD"].max()
tr_min = min(top["ECB"].min(), top["JGB10"].min(), top["DGS10"].min())
tr_max = max(top["ECB"].max(), top["JGB10"].max(), top["DGS10"].max())

# JPY 占汇率区上半 [0.77, 0.97]，EUR 占汇率区下半 [0.56, 0.75]
ax_t1.set_ylim(*zone_ylim(j_min,  j_max,  0.55, 0.88))  # JPY 缩尺
ax_t2.set_ylim(*zone_ylim(e_min,  e_max,  0.30, 0.62))  # EUR 缩尺
# 利率区 [0.02, 0.36]（上限下拉）
ax_t3.set_ylim(*zone_ylim(tr_min, tr_max, 0.02, 0.36))

# ── 利率对0线填充（ECB、JGB10）──────────────────────────
ax_t3.fill_between(t_dates, top["ECB"],   0,
                   where=(top["ECB"]   >= 0), color=C_ECB, alpha=0.18, linewidth=0)
ax_t3.fill_between(t_dates, top["ECB"],   0,
                   where=(top["ECB"]   <  0), color=C_ECB, alpha=0.18, linewidth=0)
ax_t3.fill_between(t_dates, top["JGB10"], 0,
                   where=(top["JGB10"] >= 0), color=C_JGB, alpha=0.13, linewidth=0)
ax_t3.fill_between(t_dates, top["JGB10"], 0,
                   where=(top["JGB10"] <  0), color=C_JGB, alpha=0.13, linewidth=0)

# ── 利率0线 ─────────────────────────────────────────────
ax_t3.axhline(0, color="#888888", lw=1.0, ls="--", alpha=0.7, zorder=5)
ax_t3.annotate("0%", xy=(t_dates.iloc[-1], 0),
               xytext=(6, 0), textcoords="offset points",
               fontsize=13, color="#666666", va="center")

# ── 汇率区与利率区分隔线（axes坐标 y=0.50）───────────────
from matplotlib.lines import Line2D as _L2D
sep = _L2D([0, 1], [0.50, 0.50], transform=ax_t1.transAxes,
           color="#AAAAAA", lw=1.2, ls=":", alpha=0.7, zorder=3)
ax_t1.add_line(sep)

# ════════════════════════════════════════════════════════
# 下图 Y轴范围
# ════════════════════════════════════════════════════════
u_min, u_max = bot["USD_IDX"].min(), bot["USD_IDX"].max()
u_rng = u_max - u_min
ax_b1.set_ylim(u_min - u_rng * 0.15, u_max + u_rng * 0.4)


r_min = bot["REAL5"].dropna().min()
r_max = bot["REAL5"].dropna().max()
ax_b2.set_ylim(*zone_ylim(r_min, r_max, 0.02, 0.45))

g_min, g_max = bot["Price"].min(), bot["Price"].max()
g_rng = g_max - g_min
ax_b3.set_ylim(g_min - g_rng * 0.15, g_max + g_rng * 0.05)

v_min, v_max = bot["GoldVol"].dropna().min(), bot["GoldVol"].dropna().max()
v_rng = v_max - v_min
ax_b4.set_ylim(0 - v_rng * 0.1, v_max + v_rng * 0.4)

# 第三图 ylim
vv_min = min(bot2["VXN"].dropna().min(), bot2["VXD"].dropna().min())
vv_max = max(bot2["VXN"].dropna().max(), bot2["VXD"].dropna().max())
vv_rng = vv_max - vv_min
ax_c1.set_ylim(0 - vv_rng * 0.05, vv_max + vv_rng * 0.35)

sp_min = bot2["Spread"].dropna().min()
sp_max = bot2["Spread"].dropna().max()
sp_rng = sp_max - sp_min
ax_c2.set_ylim(sp_min - sp_rng * 0.15, sp_max + sp_rng * 0.40)

# ════════════════════════════════════════════════════════
# 事件文字标注（ylim 设好后才能定 y 坐标）
# ════════════════════════════════════════════════════════
for ax in [ax_t1, ax_b1, ax_c1]:
    y_lo, y_hi = ax.get_ylim()
    y_rng = y_hi - y_lo
    for date_str, label, color, pos in events:
        xd = pd.Timestamp(date_str)
        if pos == "top":
            y_txt = y_hi - y_rng * 0.03
        else:
            y_txt = y_lo + y_rng * 0.03
        ax.text(xd, y_txt, label, color=color,
                fontsize=12, fontweight="bold", va="center", ha="center",
                rotation=0, clip_on=True, zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          alpha=0.75, ec="none"))

# ════════════════════════════════════════════════════════
# X轴格式
# ════════════════════════════════════════════════════════
for ax, dates in [(ax_t1, t_dates), (ax_b1, b_dates), (ax_c1, c_dates)]:
    ax.set_xlim(dates.iloc[0], dates.iloc[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=15, rotation=0)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.xaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

# ════════════════════════════════════════════════════════
# 轴标签
# ════════════════════════════════════════════════════════
ax_t1.set_ylabel("美元兑日元 (USD/JPY)", fontsize=17, color=C_JPY, labelpad=8)
ax_t2.set_ylabel("欧元兑美元 (EUR/USD)", fontsize=17, color=C_EUR, labelpad=8)
ax_t3.set_ylabel("利率 (%)",             fontsize=17, color="#555", labelpad=8)

ax_b1.set_ylabel("美元指数 (DXY)",        fontsize=17, color=C_USD,  labelpad=8)
ax_b2.set_ylabel("DGS5 实际利率 (%)",     fontsize=17, color=C_REAL, labelpad=8)
ax_b3.set_ylabel("黄金价格 (USD/oz)",     fontsize=17, color=C_GOLD, labelpad=8)
ax_b4.set_ylabel("黄金ETF波动率 GVZ (%)", fontsize=17, color=C_VOL,  labelpad=8)
ax_c1.set_ylabel("股票波动率指数 (%)",     fontsize=17, color=C_VXN,  labelpad=8)
ax_c2.set_ylabel("10Y-3M利差 (%)",        fontsize=17, color=C_SPR,  labelpad=8)

for ax, c in [(ax_t1,C_JPY),(ax_t2,C_EUR),(ax_t3,"#555"),
              (ax_b1,C_USD),(ax_b2,C_REAL),(ax_b3,C_GOLD),(ax_b4,C_VOL),
              (ax_c1,C_VXN),(ax_c2,C_SPR)]:
    ax.tick_params(axis="y", labelsize=15, colors=c)

# 脊线颜色
ax_t1.spines["left"].set_color(C_JPY)
ax_t2.spines["right"].set_color(C_EUR)
ax_t3.spines["right"].set_color("#555")
ax_b1.spines["left"].set_color(C_USD)
ax_b2.spines["right"].set_color(C_REAL)
ax_b3.spines["right"].set_color(C_GOLD)
ax_b4.spines["right"].set_color(C_VOL)
ax_c1.spines["left"].set_color(C_VXN)
ax_c2.spines["right"].set_color(C_SPR)

# ════════════════════════════════════════════════════════
# 标题
# ════════════════════════════════════════════════════════
fig.suptitle("货币与利率综合图", fontsize=26, fontweight="bold", y=0.99)
ax_t1.set_title("美元兑日元 · 欧元汇率 · 欧央行利率 · 日本10年期 · 美国10年期国债收益率",
                fontsize=16, color="#444", pad=8)
ax_b1.set_title("美元指数 · DGS5实际利率 · 黄金价格 · 黄金ETF波动率 GVZ",
                fontsize=16, color="#444", pad=8)
ax_c1.set_title("纳斯达克100波动率 VXN · 道琼斯波动率 VXD · 10Y-3M美债利差",
                fontsize=16, color="#444", pad=8)

# ════════════════════════════════════════════════════════
# 图例
# ════════════════════════════════════════════════════════
top_handles = [
    Line2D([0],[0], color=C_JPY,  lw=2,   label="美元兑日元 (USD/JPY)"),
    Line2D([0],[0], color=C_EUR,  lw=2,   label="欧元兑美元 (EUR/USD)"),
    Line2D([0],[0], color=C_ECB,  lw=2,   label="欧央行存款便利利率 (%)"),
    Line2D([0],[0], color=C_JGB,  lw=2,   label="日本10年期国债收益率 (%)"),
    Line2D([0],[0], color=C_DGS5, lw=2,   label="美国10年期国债收益率 (%)"),
]
bot_handles = [
    Line2D([0],[0], color=C_USD,  lw=2,          label="美元指数 (DXY)"),
    Line2D([0],[0], color=C_REAL, lw=2,           label="DGS5 实际利率 (%)"),
    Line2D([0],[0], color=C_GOLD, lw=2.4,         label="黄金价格 (USD/oz)"),
    Line2D([0],[0], color=C_VOL,  lw=2,  ls="--", label="黄金ETF波动率 GVZ (%)"),
]

bot2_handles = [
    Line2D([0],[0], color=C_VXN, lw=2,          label="纳斯达克100波动率 VXN (%)"),
    Line2D([0],[0], color=C_VXD, lw=2,          label="道琼斯波动率 VXD (%)"),
    Line2D([0],[0], color=C_SPR, lw=2, ls="--", label="10Y-3M利差 (%)"),
]

ax_t1.legend(handles=top_handles, loc="upper left",
             fontsize=14, frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
ax_b1.legend(handles=bot_handles, loc="upper left",
             fontsize=14, frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
ax_c1.legend(handles=bot2_handles, loc="upper left",
             fontsize=14, frameon=True, framealpha=0.9, edgecolor="#BBBBBB")

plt.tight_layout(rect=[0, 0, 0.80, 0.98])

out = BASE + "货币与利率_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✅ 已保存：{out}")
