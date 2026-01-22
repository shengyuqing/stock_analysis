# buy_signal_from_hot_theme.py
# pip install akshare pandas numpy openpyxl

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import akshare as ak


# ========= 你可以调的参数 =========
OUT_DIR = Path("output")   # 你的 hot_theme_scanner.py 输出目录
TOP_CONCEPTS = 20          # 只看前N热概念
PER_CONCEPT_STOCKS = 6     # 每个概念从Leaders取前N只
LOOKBACK_YEARS = 3         # 拉个股K线历史长度
ADJUST = "qfq"             # "qfq"/"" 复权，短线用qfq更稳
MIN_LIQUIDITY_AMT = 2e8    # 最低成交额过滤（2亿，可按你风格改）
VOL_BOOST = 1.5            # 突破策略：放量阈值
PULLBACK_VOL_SHRINK = 0.8  # 回踩策略：缩量阈值
NEAR_MA_PCT = 0.02         # 回踩接近均线范围（2%）


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
    # 兼容列名
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount"
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for c in ["open", "close", "high", "low", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["close"])


def build_signals(concept_rank: pd.DataFrame, leaders: pd.DataFrame) -> pd.DataFrame:
    # 1) 题材过滤：Top + 升温 + 有资金
    cr = concept_rank.copy()
    cr["last_date"] = pd.to_datetime(cr["last_date"])
    cr = cr.sort_values("score", ascending=False).head(TOP_CONCEPTS)

    cr["accel_z"] = (cr["accel"] - cr["accel"].mean()) / (cr["accel"].std() + 1e-9)
    cr = cr[(cr["amt_ratio"] > 0) & (cr["accel_z"] > 0)]  # 你也可以放宽

    hot_concepts = set(cr["concept_name"].astype(str).tolist())

    ld = leaders.copy()
    ld = ld[ld["concept_name"].astype(str).isin(hot_concepts)].copy()

    # 每个概念取前N个（Leaders本来就近似按强度排序，但这里再保险）
    ld["rank_in_concept"] = ld.groupby("concept_name")["pct_chg"].rank(ascending=False, method="first")
    ld = ld[ld["rank_in_concept"] <= PER_CONCEPT_STOCKS].copy()

    rows = []

    for r in ld.itertuples(index=False):
        code = str(r.stock_code).zfill(6)
        try:
            k = fetch_stock_daily(code, years=LOOKBACK_YEARS, adjust=ADJUST)
            if len(k) < 80:
                continue

            # 指标
            k["ma10"] = k["close"].rolling(10).mean()
            k["ma20"] = k["close"].rolling(20).mean()
            k["ma60"] = k["close"].rolling(60).mean()
            k["vol20"] = k["volume"].rolling(20).mean()
            k["atr14"] = atr14(k)

            today = k.iloc[-1]
            prev = k.iloc[-2]
            # 过去20日（不含今天）最高价/最高收盘
            high20 = k["high"].iloc[-21:-1].max()
            close20 = k["close"].iloc[-21:-1].max()

            # 流动性过滤（用“今日成交额”粗过滤，避免太小盘）
            if np.isnan(today["amount"]) or today["amount"] < MIN_LIQUIDITY_AMT:
                continue

            # ---------- 信号1：突破确认 ----------
            breakout_cond = (
                (today["close"] > today["ma20"]) and
                ((today["close"] > close20) or (today["high"] > high20)) and
                (today["volume"] > VOL_BOOST * today["vol20"])
            )

            # ---------- 信号2：强势回踩承接 ----------
            near_ma10 = abs(today["close"] - today["ma10"]) / today["ma10"] < NEAR_MA_PCT
            near_ma20 = abs(today["close"] - today["ma20"]) / today["ma20"] < NEAR_MA_PCT
            pullback_cond = (
                (today["close"] > today["ma20"]) and
                (today["ma20"] >= today["ma60"] * 0.98) and
                (near_ma10 or near_ma20) and
                (today["volume"] < PULLBACK_VOL_SHRINK * today["vol20"]) and
                (today["close"] >= today["open"])  # 简化的止跌确认
            )

            if not (breakout_cond or pullback_cond):
                continue

            # 买入触发价与止损
            if breakout_cond:
                signal_type = "突破确认"
                trigger = float(max(high20, today["high"]))
                stop = float(min(today["low"], today["ma20"] - 1.0 * (today["atr14"] if not np.isnan(today["atr14"]) else 0)))
            else:
                signal_type = "回踩承接"
                trigger = float(today["high"])   # 次日突破昨日高点再进
                stop = float(min(today["low"], today["ma20"] - 1.0 * (today["atr14"] if not np.isnan(today["atr14"]) else 0)))

            # 风险距离
            risk_pct = (trigger - stop) / trigger if trigger > 0 else np.nan

            rows.append({
                "concept": r.concept_name,
                "concept_score": float(r.concept_score),
                "code": code,
                "name": r.stock_name,
                "signal": signal_type,
                "today_date": today["date"].date(),
                "today_close": float(today["close"]),
                "today_amt": float(today["amount"]),
                "trigger_price": trigger,
                "stop_price": stop,
                "risk_pct": float(risk_pct),
                "pct_chg_today": float(r.pct_chg) if not pd.isna(r.pct_chg) else np.nan,
            })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "concept","concept_score","code","name","signal","today_date","today_close","today_amt",
            "trigger_price","stop_price","risk_pct","pct_chg_today"
        ])

    out = pd.DataFrame(rows)
    # 排序：优先题材更热 + 风险更可控
    out = out.sort_values(["concept_score", "risk_pct"], ascending=[False, True]).reset_index(drop=True)
    return out


def main():
    concept_csv = latest_file("concept_rank_*.csv")
    leaders_xlsx = latest_file("top_concepts_leaders_*.xlsx")

    concept_rank = pd.read_csv(concept_csv, encoding="utf-8-sig")
    leaders = pd.read_excel(leaders_xlsx, sheet_name="Leaders")

    signals = build_signals(concept_rank, leaders)

    save_path = OUT_DIR / f"buy_signals_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    signals.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("✅ 使用文件：")
    print(" -", concept_csv)
    print(" -", leaders_xlsx)
    print("\n✅ 输出信号：", save_path)
    if len(signals) == 0:
        print("本次没有满足条件的买入触发候选（这很正常，宁缺毋滥）。")
    else:
        print(signals.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
