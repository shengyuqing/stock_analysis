# buy_signal_from_hot_theme_v2.py
# pip install akshare pandas numpy openpyxl

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import akshare as ak


# ========= 你可以调的参数（建议先用默认）=========
OUT_DIR = Path("output")

# 热题材数量（建议 6~10，不要20那么大）
HOT_CONCEPTS_N = 8

# 每个概念先从 Leaders 取多少只候选（建议 2~4）
PER_CONCEPT_STOCKS = 3

# 最终每个题材最多保留几只
MAX_FINAL_PER_CONCEPT = 2

# 最终总输出上限
MAX_SIGNALS_TOTAL = 20

# 去重：同一只股票如果被多个题材选中，只保留“综合评分最高”的那条
DEDUP_BY_STOCK = True

LOOKBACK_YEARS = 3
ADJUST = "qfq"

# 流动性过滤（成交额，单位：元；2e8=2亿）
MIN_LIQUIDITY_AMT = 2e8

# 市场环境过滤：HS300 在 MA60 之上才放开做突破；否则只做回踩
USE_MARKET_FILTER = True

# 技术过滤：不过度拉伸（距离MA20不能太远）
MAX_EXTEND_FROM_MA20 = 0.12  # 12%

# 突破放量阈值
VOL_BOOST = 1.5

# 回踩缩量阈值
PULLBACK_VOL_SHRINK = 0.8

# 回踩接近均线范围
NEAR_MA_PCT = 0.02

# 风险过滤：止损距离太小/太大都不要（避免噪声/避免不可控）
RISK_PCT_MIN = 0.02   # 2%
RISK_PCT_MAX = 0.08   # 8%


def latest_file(pattern: str) -> Path:
    files = sorted(OUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"在 {OUT_DIR} 下找不到 {pattern}")
    return files[0]


def atr14(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def fetch_stock_daily(code: str, years: int = 3, adjust: str = "qfq") -> pd.DataFrame:
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=365 * years)

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust=adjust
    )
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount"
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"])


def fetch_hs300_daily(years: int = 3) -> pd.DataFrame:
    """
    沪深300指数（用于市场环境过滤）
    """
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=365 * years)
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df = df.rename(columns={"date": "date", "open": "open", "close": "close", "high": "high", "low": "low", "volume": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date").reset_index(drop=True)
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"])


def calc_market_regime_ok(years: int = 3) -> bool:
    if not USE_MARKET_FILTER:
        return True
    try:
        idx = fetch_hs300_daily(years=years)
        if len(idx) < 80:
            return True
        idx["ma60"] = idx["close"].rolling(60).mean()
        today = idx.iloc[-1]
        return bool(today["close"] > today["ma60"])
    except Exception:
        # 指数取不到就不阻断
        return True


def build_hot_concepts(concept_rank: pd.DataFrame) -> set[str]:
    """
    更严谨的题材筛选：
    - 不在 TOP20 里算 accel_z（样本太小不稳定）
    - 在全体概念上算 accel_z / amt_ratio_z
    - 再从 score 前 80 里筛出真强题材，最后取前 HOT_CONCEPTS_N
    """
    cr = concept_rank.copy()
    cr["last_date"] = pd.to_datetime(cr["last_date"])

    # 在全体上做 zscore（避免你遇到 accel 全负就全被筛掉）
    cr["accel_z"] = (cr["accel"] - cr["accel"].mean()) / (cr["accel"].std() + 1e-9)
    cr["amt_z"] = (cr["amt_ratio"] - cr["amt_ratio"].mean()) / (cr["amt_ratio"].std() + 1e-9)

    # 先拿大池子（避免错杀）
    cr = cr.sort_values("score", ascending=False).head(80)

    # 真强题材条件（你可以调整）
    # 1) 资金不弱：amt_ratio > 0（或 amt_z > -0.2）
    # 2) 加速不要求>0，但要“相对不差”：accel_z > -0.3
    # 3) 趋势确认：breakout_60d=1 或 ret_5d>0
    cr = cr[
        ((cr["amt_ratio"] > 0) | (cr["amt_z"] > -0.2)) &
        (cr["accel_z"] > -0.3) &
        ((cr["breakout_60d"] >= 1) | (cr["ret_5d"] > 0))
    ]

    # 最后取最热的 N 个
    cr = cr.sort_values("score", ascending=False).head(HOT_CONCEPTS_N)
    return set(cr["concept_name"].astype(str).tolist())


def build_signals(concept_rank: pd.DataFrame, leaders: pd.DataFrame) -> pd.DataFrame:
    market_ok_for_breakout = calc_market_regime_ok(years=LOOKBACK_YEARS)

    hot_concepts = build_hot_concepts(concept_rank)

    ld = leaders.copy()
    ld["stock_code"] = ld["stock_code"].astype(str).str.zfill(6)
    ld = ld[ld["concept_name"].astype(str).isin(hot_concepts)].copy()

    # 每个概念只取前 PER_CONCEPT_STOCKS 个（候选池变小）
    ld["rank_in_concept"] = ld.groupby("concept_name")["pct_chg"].rank(ascending=False, method="first")
    ld = ld[ld["rank_in_concept"] <= PER_CONCEPT_STOCKS].copy()

    rows = []

    for r in ld.itertuples(index=False):
        code = str(r.stock_code).zfill(6)
        name = str(r.stock_name)

        # 基础过滤：ST 直接不要（你可以自行放宽）
        if "ST" in name.upper():
            continue

        try:
            k = fetch_stock_daily(code, years=LOOKBACK_YEARS, adjust=ADJUST)
            if len(k) < 120:
                continue

            # 指标
            k["ma10"] = k["close"].rolling(10).mean()
            k["ma20"] = k["close"].rolling(20).mean()
            k["ma60"] = k["close"].rolling(60).mean()
            k["vol20"] = k["volume"].rolling(20).mean()
            k["atr14"] = atr14(k)

            today = k.iloc[-1]
            prev = k.iloc[-2]

            # 流动性过滤
            if np.isnan(today["amount"]) or today["amount"] < MIN_LIQUIDITY_AMT:
                continue

            # 趋势过滤：强趋势（更严谨）
            ma20_slope = (k["ma20"].iloc[-1] / (k["ma20"].iloc[-6] + 1e-12)) - 1.0
            ma60_slope = (k["ma60"].iloc[-1] / (k["ma60"].iloc[-11] + 1e-12)) - 1.0
            trend_ok = (today["close"] > today["ma20"]) and (today["ma20"] > today["ma60"]) and (ma20_slope > 0) and (ma60_slope > -0.002)

            if not trend_ok:
                continue

            # 不过度拉伸：离MA20太远的不要（防追高）
            extend = (today["close"] / (today["ma20"] + 1e-12)) - 1.0
            if extend > MAX_EXTEND_FROM_MA20:
                continue

            # 相对强度：近20日收益 -（用自身近20日均线斜率替代指数也行，这里用收益）
            ret20 = (k["close"].iloc[-1] / (k["close"].iloc[-21] + 1e-12)) - 1.0
            # 简化：要求 ret20 > 0，避免弱票混进来
            if ret20 <= 0:
                continue

            # 过去20日区间高点
            high20 = k["high"].iloc[-21:-1].max()
            close20 = k["close"].iloc[-21:-1].max()

            vol_ratio = today["volume"] / (today["vol20"] + 1e-12)

            # ---------- 信号1：突破确认（更严格） ----------
            breakout_cond = (
                market_ok_for_breakout and
                (today["close"] > close20) and
                (vol_ratio > VOL_BOOST)
            )

            # ---------- 信号2：强势回踩承接（更严格） ----------
            near_ma10 = abs(today["close"] - today["ma10"]) / (today["ma10"] + 1e-12) < NEAR_MA_PCT
            near_ma20 = abs(today["close"] - today["ma20"]) / (today["ma20"] + 1e-12) < NEAR_MA_PCT

            # 回踩策略要确保“之前确实强过”：最近10天内出现过接近20日新高（主升过）
            recent10_high = k["high"].iloc[-11:-1].max()
            had_strength = recent10_high >= high20 * 0.98

            pullback_cond = (
                (near_ma10 or near_ma20) and
                had_strength and
                (vol_ratio < PULLBACK_VOL_SHRINK) and
                (today["close"] >= today["open"]) and
                (today["low"] >= k["low"].iloc[-6:-1].min() * 0.98)  # 不再加速破位
            )

            if not (breakout_cond or pullback_cond):
                continue

            # 买入触发价与止损（ATR 防噪声）
            atr = float(today["atr14"]) if not np.isnan(today["atr14"]) else 0.0

            if breakout_cond:
                signal_type = "突破确认"
                trigger = float(max(high20, today["high"]))
                stop = float(min(today["low"], today["ma20"] - 1.2 * atr))
            else:
                signal_type = "回踩承接"
                trigger = float(today["high"])  # 次日突破昨日高点再进
                stop = float(min(today["low"], today["ma20"] - 1.0 * atr))

            if trigger <= 0:
                continue

            risk_pct = (trigger - stop) / trigger
            if not (RISK_PCT_MIN <= risk_pct <= RISK_PCT_MAX):
                continue

            # 质量评分：用于最终排序（越大越优先）
            # 你可以理解成：题材热 + 趋势强 + 量价到位 + 风险可控
            quality = (
                0.45 * float(r.concept_score) +
                0.25 * float(ret20) * 10 +         # ret20 放大一些让它有区分度
                0.20 * float(vol_ratio) +
                0.10 * (1.0 - float(risk_pct))
            )

            rows.append({
                "concept": str(r.concept_name),
                "concept_score": float(r.concept_score),
                "code": code,
                "name": name,
                "signal": signal_type,
                "today_date": today["date"].date(),
                "today_close": float(today["close"]),
                "today_amt": float(today["amount"]),
                "vol_ratio": float(vol_ratio),
                "ret20": float(ret20),
                "trigger_price": trigger,
                "stop_price": stop,
                "risk_pct": float(risk_pct),
                "quality": float(quality),
                "pct_chg_today": float(r.pct_chg) if not pd.isna(r.pct_chg) else np.nan,
            })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "concept","concept_score","code","name","signal","today_date","today_close","today_amt",
            "vol_ratio","ret20","trigger_price","stop_price","risk_pct","quality","pct_chg_today"
        ])

    out = pd.DataFrame(rows)

    # ✅ 排序：先看质量分，再看题材热度，再看风险
    out = out.sort_values(["quality", "concept_score", "risk_pct"], ascending=[False, False, True]).reset_index(drop=True)

    # ✅ 同股去重（避免你看到同一批票被多个题材重复刷屏）
    if DEDUP_BY_STOCK:
        out = out.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)

    # ✅ 每题材最多保留 MAX_FINAL_PER_CONCEPT 只
    out["rank_final"] = out.groupby("concept").cumcount() + 1
    out = out[out["rank_final"] <= MAX_FINAL_PER_CONCEPT].drop(columns=["rank_final"]).reset_index(drop=True)

    # ✅ 全局最多保留 MAX_SIGNALS_TOTAL 只
    out = out.head(MAX_SIGNALS_TOTAL).reset_index(drop=True)

    return out


def main():
    concept_csv = latest_file("concept_rank_*.csv")
    leaders_xlsx = latest_file("top_concepts_leaders_*.xlsx")

    concept_rank = pd.read_csv(concept_csv, encoding="utf-8-sig")
    leaders = pd.read_excel(leaders_xlsx, sheet_name="Leaders")

    signals = build_signals(concept_rank, leaders)

    save_path = OUT_DIR / f"buy_signals_v2_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    signals.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("✅ 使用文件：")
    print(" -", concept_csv)
    print(" -", leaders_xlsx)
    print("✅ 输出：", save_path)

    if len(signals) == 0:
        print("本次没有满足更严苛条件的候选（正常，宁缺毋滥）。")
    else:
        print(signals.to_string(index=False))


if __name__ == "__main__":
    main()
