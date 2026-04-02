"""
贵金属价格与波动率（2×2 矩阵布局）
  左上 : 黄金价格  — 含自动识别的收敛三角形
  右上 : 白银价格  — 含自动识别的收敛三角形
  左下 : 黄金波动率 GVZ
  右下 : 白银波动率 VXSLV（截至 2022-02）

  绿色阴影 : 波动率持续低迷（低于各自 30th pct）超过 1 年的区间
  紫色三角 : 收敛三角形（对称/上升/下降均识别）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BASE = "/Users/laikaiyuan/Downloads/读书/黄金大师课/Gold Price/"

# ════════════════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════════════════

def load_investing(fname):
    df = pd.read_csv(BASE + fname, quotechar='"', skipinitialspace=True)
    df.columns = df.columns.str.strip().str.replace('"', '')
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(
        df["Price"].astype(str).str.replace(",", ""), errors="coerce")
    return df[["Date", "Price"]].dropna().sort_values("Date").reset_index(drop=True)

xau = load_investing("XAU_USD weekly Historical Data.csv")
xag = load_investing("XAG_USD weekly Historical Data.csv")

gvz = pd.read_csv(BASE + "CBOE黄金ETF波动率指数.csv")
gvz["Date"] = pd.to_datetime(gvz["observation_date"])
gvz = gvz[["Date", "GVZCLS"]].rename(columns={"GVZCLS": "Vol"}).dropna().sort_values("Date")

vxslv = pd.read_csv(BASE + "CBOE白银ETF波动率指数.csv")
vxslv["Date"] = pd.to_datetime(vxslv["observation_date"])
vxslv = vxslv[["Date", "VXSLVCLS"]].rename(columns={"VXSLVCLS": "Vol"}).dropna().sort_values("Date")

START = pd.Timestamp("2008-06-01")
END   = xau["Date"].max()

for df in [xau, xag, gvz, vxslv]:
    df.drop(df[df["Date"] < START].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

# ════════════════════════════════════════════════════════
# 2. 识别"低波动率超过 1 年"的区间
# ════════════════════════════════════════════════════════
LOW_PCTL = 30
MIN_DAYS = 365

def find_low_vol_periods(vol_df, pctl=LOW_PCTL, min_days=MIN_DAYS):
    threshold = np.percentile(vol_df["Vol"], pctl)
    is_low = vol_df["Vol"] < threshold
    periods, in_period, seg_start = [], False, None
    for date, low in zip(vol_df["Date"], is_low):
        if low and not in_period:
            in_period, seg_start = True, date
        elif not low and in_period:
            in_period = False
            if (date - seg_start).days >= min_days:
                periods.append((seg_start, date))
    if in_period and (vol_df["Date"].iloc[-1] - seg_start).days >= min_days:
        periods.append((seg_start, vol_df["Date"].iloc[-1]))
    return threshold, periods

gvz_thresh,   gvz_periods   = find_low_vol_periods(gvz)
vxslv_thresh, vxslv_periods = find_low_vol_periods(vxslv)

# ════════════════════════════════════════════════════════
# 3. 收敛三角形识别
# ════════════════════════════════════════════════════════

def find_triangles(dates_s, prices_s, min_span=20, max_span=260, max_count=3):
    prices_np  = prices_s.values.astype(float)
    dates_num  = mdates.date2num(pd.to_datetime(dates_s.values))
    price_rng  = np.nanmax(prices_np) - np.nanmin(prices_np)

    peaks,   _ = find_peaks( prices_np, prominence=price_rng * 0.025, distance=3)
    troughs, _ = find_peaks(-prices_np, prominence=price_rng * 0.025, distance=3)

    triangles = []

    for i in range(len(peaks) - 1):
        for j in range(i + 1, len(peaks)):
            s, e = peaks[i], peaks[j]
            span = e - s
            if not (min_span <= span <= max_span):
                continue

            win_peaks   = peaks[(peaks >= s) & (peaks <= e)]
            win_troughs = troughs[(troughs >= s) & (troughs <= e)]
            if len(win_peaks) < 2 or len(win_troughs) < 2:
                continue

            xp = dates_num[win_peaks]
            yp = prices_np[win_peaks]
            up_sl, up_int = np.polyfit(xp, yp, 1)

            xt = dates_num[win_troughs]
            yt = prices_np[win_troughs]
            lo_sl, lo_int = np.polyfit(xt, yt, 1)

            if up_sl >= lo_sl:
                continue

            apex_x = (lo_int - up_int) / (up_sl - lo_sl)
            ahead  = apex_x - dates_num[e]
            if ahead < 0 or ahead > 4 * 365:
                continue

            mid_p = np.mean(prices_np[s:e + 1])
            res   = (np.std(yp - (up_sl * xp + up_int)) +
                     np.std(yt - (lo_sl * xt + lo_int))) / mid_p
            score = span / (1 + res * 100)

            triangles.append(dict(
                start=s, end=e,
                up_sl=up_sl, up_int=up_int,
                lo_sl=lo_sl, lo_int=lo_int,
                apex_x=apex_x, score=score,
            ))

    triangles.sort(key=lambda x: -x["score"])
    kept = []
    for tri in triangles:
        if not any(tri["start"] < k["end"] and tri["end"] > k["start"] for k in kept):
            kept.append(tri)
        if len(kept) >= max_count:
            break

    print(f"  识别到 {len(kept)} 个收敛三角形")
    for k in kept:
        d_s = mdates.num2date(dates_num[k["start"]]).strftime("%Y-%m")
        d_e = mdates.num2date(dates_num[k["end"]]).strftime("%Y-%m")
        print(f"    {d_s} → {d_e}  上轨斜率={k['up_sl']:.4f}  下轨斜率={k['lo_sl']:.4f}")

    return kept, dates_num, prices_np


def draw_triangles(ax, triangles, dates_num, prices_np,
                   color="#6A1B9A", label="收敛三角形"):
    for tri in triangles:
        # 跨度 = 低波动区间本身（x_start / x_end 存的是绝对日期数值）
        x_s = tri.get("x_start", dates_num[tri["start"]])
        x_e = tri.get("x_end",   dates_num[tri["end"]])

        x_vals = np.linspace(x_s, x_e, 400)
        y_up   = tri["up_sl"] * x_vals + tri["up_int"]
        y_lo   = tri["lo_sl"] * x_vals + tri["lo_int"]

        x_dates = [mdates.num2date(x) for x in x_vals]

        ax.plot(x_dates, y_up, color=color, lw=2.2, ls="-", zorder=7, alpha=0.9,
                label=label)
        ax.plot(x_dates, y_lo, color=color, lw=2.2, ls="-", zorder=7, alpha=0.9)
        ax.fill_between(x_dates, y_up, y_lo,
                        color=color, alpha=0.10, zorder=2)

        mid_x = (x_s + x_e) / 2
        mid_y = ((tri["up_sl"] * mid_x + tri["up_int"]) +
                 (tri["lo_sl"] * mid_x + tri["lo_int"])) / 2
        ax.text(mdates.num2date(mid_x), mid_y, label,
                ha="center", va="center", fontsize=12,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          alpha=0.85, ec=color, lw=0.8), zorder=8)


# 只保留 2017-04-19 → 2019-06-20 的低波动区间（唯一超 1 年的区间）
LOW_VOL_START = pd.Timestamp("2017-04-19")
LOW_VOL_END   = pd.Timestamp("2019-06-20")

def find_triangle_in_window(dates_s, prices_s, win_start, win_end):
    """在指定时间窗口内拟合一个收敛三角形，跨度固定为 [win_start, win_end]"""
    mask = (dates_s >= win_start) & (dates_s <= win_end)
    d_w  = dates_s[mask].reset_index(drop=True)
    p_w  = prices_s[mask].reset_index(drop=True)

    prices_np = p_w.values.astype(float)
    dates_num = mdates.date2num(pd.to_datetime(d_w.values))
    price_rng = np.nanmax(prices_np) - np.nanmin(prices_np)

    peaks,   _ = find_peaks( prices_np, prominence=price_rng * 0.02, distance=3)
    troughs, _ = find_peaks(-prices_np, prominence=price_rng * 0.02, distance=3)

    if len(peaks) < 2 or len(troughs) < 2:
        print("  峰/谷点不足，无法拟合")
        return None, dates_num, prices_np

    xp = dates_num[peaks];  yp = prices_np[peaks]
    xt = dates_num[troughs]; yt = prices_np[troughs]
    up_sl, up_int = np.polyfit(xp, yp, 1)
    lo_sl, lo_int = np.polyfit(xt, yt, 1)

    # 顶点
    if up_sl < lo_sl:
        apex_x = (lo_int - up_int) / (up_sl - lo_sl)
    else:
        apex_x = dates_num[-1]   # 不收敛则延到末尾

    tri = dict(
        start=0, end=len(prices_np) - 1,
        up_sl=up_sl, up_int=up_int,
        lo_sl=lo_sl, lo_int=lo_int,
        apex_x=apex_x,
        # 直接存绝对日期数值（覆盖索引用）
        x_start=dates_num[0],
        x_end=dates_num[-1],
    )
    print(f"  上轨斜率={up_sl:.4f}  下轨斜率={lo_sl:.4f}")
    return tri, dates_num, prices_np


print("黄金三角形（2017-04 → 2019-06）:")
xau_tri_w, xau_dnum, xau_np = find_triangle_in_window(
    xau["Date"], xau["Price"], LOW_VOL_START, LOW_VOL_END)
xau_tri = [xau_tri_w] if xau_tri_w else []

print("白银三角形（2017-04 → 2019-06）:")
xag_tri_w, xag_dnum, xag_np = find_triangle_in_window(
    xag["Date"], xag["Price"], LOW_VOL_START, LOW_VOL_END)
xag_tri = [xag_tri_w] if xag_tri_w else []

# ════════════════════════════════════════════════════════
# 4. 颜色 & 布局
# ════════════════════════════════════════════════════════
C_GOLD   = "#D4A017"
C_SILVER = "#8E9DAE"
C_GVZ    = "#C62828"
C_VXSLV  = "#1565C0"
C_SHADE  = "#4CAF50"
C_TRI    = "#6A1B9A"
BG       = "#FAFAF8"
lw       = 1.8

fig, axes = plt.subplots(2, 2, figsize=(28, 16),
                          gridspec_kw={"hspace": 0.12, "wspace": 0.10})
fig.patch.set_facecolor(BG)

ax_gold  = axes[0, 0]
ax_silv  = axes[0, 1]
ax_gvz   = axes[1, 0]
ax_vxslv = axes[1, 1]

for ax in axes.flat:
    ax.set_facecolor(BG)

# ════════════════════════════════════════════════════════
# 5. 画图
# ════════════════════════════════════════════════════════

# ── 左上：黄金价格 ────────────────────────────────────────────────────────────
ax_gold.plot(xau["Date"], xau["Price"], color=C_GOLD, lw=lw + 0.4, zorder=4)
ax_gold.fill_between(xau["Date"], xau["Price"], xau["Price"].min() * 0.95,
                     color=C_GOLD, alpha=0.12, linewidth=0)
ax_gold.set_ylabel("USD / oz", fontsize=17, color=C_GOLD, labelpad=8)
ax_gold.tick_params(axis="y", labelsize=15, colors=C_GOLD)
ax_gold.spines["left"].set_color(C_GOLD)
ax_gold.set_title("黄金价格 (XAU/USD)", fontsize=17, color="#444", pad=8)

draw_triangles(ax_gold, xau_tri, xau_dnum, xau_np, color=C_TRI)

# ── 右上：白银价格 ────────────────────────────────────────────────────────────
ax_silv.plot(xag["Date"], xag["Price"], color=C_SILVER, lw=lw, zorder=4)
ax_silv.fill_between(xag["Date"], xag["Price"], xag["Price"].min() * 0.95,
                     color=C_SILVER, alpha=0.18, linewidth=0)
ax_silv.set_ylabel("USD / oz", fontsize=17, color=C_SILVER, labelpad=8)
ax_silv.tick_params(axis="y", labelsize=15, colors=C_SILVER)
ax_silv.spines["left"].set_color(C_SILVER)
ax_silv.set_title("白银价格 (XAG/USD)", fontsize=17, color="#444", pad=8)

draw_triangles(ax_silv, xag_tri, xag_dnum, xag_np, color=C_TRI)

# ── 左下：黄金波动率 GVZ ──────────────────────────────────────────────────────
ax_gvz.plot(gvz["Date"], gvz["Vol"], color=C_GVZ, lw=lw * 0.8, zorder=4, alpha=0.85)
ax_gvz.fill_between(gvz["Date"], gvz["Vol"], 0, color=C_GVZ, alpha=0.18, linewidth=0)
ax_gvz.axhline(gvz_thresh, color=C_GVZ, lw=1.3, ls="--", alpha=0.65,
               label=f"低波动阈值 {gvz_thresh:.1f}（第 {LOW_PCTL} 百分位）")
ax_gvz.set_ylabel("GVZ", fontsize=17, color=C_GVZ, labelpad=8)
ax_gvz.tick_params(axis="y", labelsize=15, colors=C_GVZ)
ax_gvz.spines["left"].set_color(C_GVZ)
ax_gvz.set_title("黄金波动率 GVZ（CBOE Gold ETF Volatility）", fontsize=17, color="#444", pad=8)
ax_gvz.set_ylim(bottom=0)

# ── 右下：白银波动率 VXSLV ────────────────────────────────────────────────────
ax_vxslv.plot(vxslv["Date"], vxslv["Vol"], color=C_VXSLV, lw=lw * 0.8, zorder=4, alpha=0.85)
ax_vxslv.fill_between(vxslv["Date"], vxslv["Vol"], 0, color=C_VXSLV, alpha=0.18, linewidth=0)
ax_vxslv.axhline(vxslv_thresh, color=C_VXSLV, lw=1.3, ls="--", alpha=0.65,
                 label=f"低波动阈值 {vxslv_thresh:.1f}（第 {LOW_PCTL} 百分位）")
vxslv_end = vxslv["Date"].iloc[-1]
ax_vxslv.axvline(vxslv_end, color="#888", lw=1.2, ls=":", zorder=6)
ax_vxslv.text(vxslv_end + pd.Timedelta("60d"), vxslv["Vol"].max() * 0.92,
              "数据截至\n2022-02", fontsize=13, color="#888", va="top", ha="left")
ax_vxslv.set_ylabel("VXSLV", fontsize=17, color=C_VXSLV, labelpad=8)
ax_vxslv.tick_params(axis="y", labelsize=15, colors=C_VXSLV)
ax_vxslv.spines["left"].set_color(C_VXSLV)
ax_vxslv.set_title("白银波动率 VXSLV（CBOE Silver ETF Volatility，截至 2022-02）",
                   fontsize=17, color="#444", pad=8)
ax_vxslv.set_ylim(bottom=0)

# ════════════════════════════════════════════════════════
# 6. 低波动区间阴影
# ════════════════════════════════════════════════════════
def shade_periods(ax, periods, color=C_SHADE, alpha=0.18):
    for s, e in periods:
        ax.axvspan(s, e, color=color, alpha=alpha, zorder=1)

for ax in [ax_gold, ax_gvz]:
    shade_periods(ax, gvz_periods)
for ax in [ax_silv, ax_vxslv]:
    shade_periods(ax, vxslv_periods)

# ════════════════════════════════════════════════════════
# 7. 低波动区间时长标注
# ════════════════════════════════════════════════════════
def label_periods(ax, periods, vol_df):
    y_max = vol_df["Vol"].max()
    for s, e in periods:
        mid = s + (e - s) / 2
        dur = (e - s).days / 365.25
        ax.text(mid, y_max * 0.90, f"{dur:.1f}年",
                ha="center", va="top", fontsize=14, color="#2E7D32",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="none"))

label_periods(ax_gvz,   gvz_periods,   gvz)
label_periods(ax_vxslv, vxslv_periods, vxslv)

# ════════════════════════════════════════════════════════
# 8. 图例
# ════════════════════════════════════════════════════════
shade_gvz   = mpatches.Patch(color=C_SHADE, alpha=0.4,
                              label=f"GVZ 低位 > 1年（< {gvz_thresh:.1f}）")
shade_vxslv = mpatches.Patch(color=C_SHADE, alpha=0.4,
                              label=f"VXSLV 低位 > 1年（< {vxslv_thresh:.1f}）")
tri_patch   = Line2D([0],[0], color=C_TRI, lw=2, label="收敛三角形")

ax_gold.legend(handles=[Line2D([0],[0], color=C_GOLD, lw=2, label="黄金价格"),
                         shade_gvz, tri_patch],
               fontsize=13, loc="upper left", frameon=True,
               framealpha=0.9, edgecolor="#BBBBBB")
ax_silv.legend(handles=[Line2D([0],[0], color=C_SILVER, lw=2, label="白银价格"),
                         shade_vxslv,
                         Line2D([0],[0], color=C_TRI, lw=2, label="收敛三角形")],
               fontsize=13, loc="upper left", frameon=True,
               framealpha=0.9, edgecolor="#BBBBBB")
ax_gvz.legend(handles=[
    Line2D([0],[0], color=C_GVZ, lw=1.3, ls="--", alpha=0.65,
           label=f"低波动阈值 {gvz_thresh:.1f}"),
    shade_gvz],
    fontsize=13, loc="upper left", frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
ax_vxslv.legend(handles=[
    Line2D([0],[0], color=C_VXSLV, lw=1.3, ls="--", alpha=0.65,
           label=f"低波动阈值 {vxslv_thresh:.1f}"),
    shade_vxslv],
    fontsize=13, loc="upper left", frameon=True, framealpha=0.9, edgecolor="#BBBBBB")

# ════════════════════════════════════════════════════════
# 9. X 轴格式
# ════════════════════════════════════════════════════════
for ax in axes.flat:
    ax.set_xlim(START, END)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=15, rotation=0)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.6)
    ax.xaxis.grid(True, color="#CCCCCC", lw=0.5, alpha=0.3)
    ax.set_axisbelow(True)

for ax in [ax_gold, ax_silv]:
    ax.tick_params(labelbottom=False)

# ════════════════════════════════════════════════════════
# 10. 大标题 & 保存
# ════════════════════════════════════════════════════════
plt.tight_layout()
fig.suptitle("贵金属价格与波动率分析（黄金 · 白银）",
             fontsize=26, fontweight="bold", y=1.0)
plt.subplots_adjust(top=0.952)

out = BASE + "贵金属价格与波动率_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n✅ 已保存：{out}")
plt.show()
