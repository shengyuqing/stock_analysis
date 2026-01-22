# hot_theme_swing_only.py
# pip install -U akshare pandas numpy openpyxl

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path
import time
import random
import inspect
import numpy as np
import pandas as pd
import akshare as ak


# =========================
# 配置（你主要调这几个就行）
# =========================
@dataclass
class Config:
    # 概念指数历史区间（避免默认 end_date=20221128）
    concept_years: int = 5
    concept_lookback: int = 260 * 5

    # 个股历史只需要近两年足够算 MA/ATR
    stock_days: int = 520

    # 取前多少个题材做候选池
    top_k_concepts: int = 30

    # 每个题材先按成交额选前N只股票再做形态判断（避免挑到一堆涨停强势不回踩的）
    preselect_top_by_amount: int = 40

    # 最终每个题材输出几只波段候选
    swing_per_concept: int = 3

    # 并发与限流
    max_workers_concept: int = 4
    max_workers_stock: int = 4
    retries: int = 2
    sleep_range: tuple[float, float] = (0.05, 0.20)

    # 波段筛选阈值（空的话就放宽这些）
    min_amount: float = 2e8          # 成交额门槛（2亿）
    pullback_tol: float = 0.03       # 回踩MA20容忍度（3%）
    vol_contract: float = 1.05       # 缩量阈值：vol <= 1.05 * vol_ma20（稍微放松，避免空）
    require_vol_contract: bool = False  # 是否强制要求缩量（建议先 False，跑通后再 True）

    # 触发价与止损
    trigger_buffer: float = 0.002    # 明日突破触发：* (1+0.2%)
    stop_low_buffer: float = 0.01    # 止损：今日低点下方 1%
    stop_ma60_buffer: float = 0.02   # 止损：MA60 下方 2%
    atr_mult: float = 1.2            # ATR 止损倍数

    out_dir: str = "output"

    # 调试：>0 只跑前N个概念，先验证逻辑/速度
    debug_sample: int = 0


# =========================
# 工具函数
# =========================
def now_ymd() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d")


def _pick_col(df: pd.DataFrame, keywords: list[str], fallback_idx: int | None = None) -> str:
    cols = list(df.columns)
    for k in keywords:
        for c in cols:
            if k == c:
                return c
    for k in keywords:
        for c in cols:
            if k in str(c):
                return c
    if fallback_idx is not None:
        return cols[fallback_idx]
    raise KeyError(f"找不到列：keywords={keywords}，现有列={cols}")


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return x * 0
    return (x - mu) / sd


def call_with_supported_params(func, **kwargs):
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return func(**filtered)


def normalize_concept_hist(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_col(df, ["日期", "date"])
    close_col = None
    for keys in (["收盘", "close"], ["收盘价"], ["最新价"]):
        try:
            close_col = _pick_col(df, keys)
            break
        except Exception:
            pass
    if close_col is None:
        raise KeyError("概念历史：找不到收盘列")

    vol_col = None
    amt_col = None
    try:
        vol_col = _pick_col(df, ["成交量", "volume"])
    except Exception:
        vol_col = None
    try:
        amt_col = _pick_col(df, ["成交额", "amount"])
    except Exception:
        amt_col = None
    if vol_col is None and amt_col is None:
        raise KeyError("概念历史：找不到成交量/成交额列")
    if vol_col is None:
        vol_col = amt_col
    if amt_col is None:
        amt_col = vol_col

    out = df[[date_col, close_col, vol_col, amt_col]].copy()
    out.columns = ["date", "close", "volume", "amount"]
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date")
    return out


def normalize_stock_hist(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_col(df, ["日期", "date"])
    open_col = _pick_col(df, ["开盘", "open"])
    high_col = _pick_col(df, ["最高", "high"])
    low_col = _pick_col(df, ["最低", "low"])
    close_col = _pick_col(df, ["收盘", "close"])

    vol_col = None
    amt_col = None
    try:
        vol_col = _pick_col(df, ["成交量", "volume"])
    except Exception:
        vol_col = None
    try:
        amt_col = _pick_col(df, ["成交额", "amount"])
    except Exception:
        amt_col = None
    if vol_col is None and amt_col is None:
        raise KeyError("个股历史：找不到成交量/成交额列")
    if vol_col is None:
        vol_col = amt_col
    if amt_col is None:
        amt_col = vol_col

    out = df[[date_col, open_col, high_col, low_col, close_col, vol_col, amt_col]].copy()
    out.columns = ["date", "open", "high", "low", "close", "volume", "amount"]
    out["date"] = pd.to_datetime(out["date"])
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values("date").drop_duplicates("date")
    return out


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()


# =========================
# 数据拉取
# =========================
def fetch_concept_list() -> pd.DataFrame:
    df = ak.stock_board_concept_name_em()
    code_col = _pick_col(df, ["板块代码", "代码", "symbol"])
    name_col = _pick_col(df, ["板块名称", "名称", "name"])
    out = df[[code_col, name_col]].rename(columns={code_col: "concept_code", name_col: "concept_name"})
    out["concept_code"] = out["concept_code"].astype(str)
    out["concept_name"] = out["concept_name"].astype(str)
    return out


def concept_start_end(cfg: Config) -> tuple[str, str]:
    end = pd.Timestamp.now().strftime("%Y%m%d")
    start = (pd.Timestamp.now() - pd.Timedelta(days=365 * cfg.concept_years)).strftime("%Y%m%d")
    return start, end


def stock_start_end(cfg: Config) -> tuple[str, str]:
    end = pd.Timestamp.now().strftime("%Y%m%d")
    start = (pd.Timestamp.now() - pd.Timedelta(days=cfg.stock_days)).strftime("%Y%m%d")
    return start, end


def fetch_concept_hist(concept_name: str, concept_code: str, cfg: Config) -> pd.DataFrame:
    start, end = concept_start_end(cfg)
    last_err = None

    for sym in [concept_name, concept_code]:
        for _ in range(cfg.retries + 1):
            try:
                time.sleep(random.uniform(*cfg.sleep_range))
                df = call_with_supported_params(
                    ak.stock_board_concept_hist_em,
                    symbol=sym, period="daily", start_date=start, end_date=end, adjust=""
                )
                if isinstance(df, pd.DataFrame) and len(df) >= 60:
                    return normalize_concept_hist(df)
            except Exception as e:
                last_err = e

    # fallback 同花顺概念指数历史
    for _ in range(cfg.retries + 1):
        try:
            time.sleep(random.uniform(*cfg.sleep_range))
            df = call_with_supported_params(
                ak.stock_board_concept_hist_ths,
                symbol=concept_name, start_date=start, end_date=end
            )
            if isinstance(df, pd.DataFrame) and len(df) >= 60:
                return normalize_concept_hist(df)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"概念历史拉取失败：{concept_name}/{concept_code} last_err={last_err}")


def fetch_concept_cons(concept_name: str, concept_code: str, cfg: Config) -> pd.DataFrame:
    last_err = None
    for sym in [concept_name, concept_code]:
        for _ in range(cfg.retries + 1):
            try:
                time.sleep(random.uniform(*cfg.sleep_range))
                df = call_with_supported_params(ak.stock_board_concept_cons_em, symbol=sym)
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"概念成分拉取失败：{concept_name}/{concept_code} last_err={last_err}")


def fetch_a_spot() -> pd.DataFrame:
    spot = ak.stock_zh_a_spot_em()
    code_col = _pick_col(spot, ["代码", "证券代码"])
    name_col = _pick_col(spot, ["名称", "证券名称"])
    amt_col = _pick_col(spot, ["成交额"])
    pct_col = _pick_col(spot, ["涨跌幅"])

    spot = spot[[code_col, name_col, amt_col, pct_col]].copy()
    spot.columns = ["code", "name", "amount", "pct_chg"]
    spot["code"] = spot["code"].astype(str).str.zfill(6)
    spot["amount"] = pd.to_numeric(spot["amount"], errors="coerce")
    spot["pct_chg"] = pd.to_numeric(spot["pct_chg"], errors="coerce")
    return spot


def fetch_stock_hist(code: str, cfg: Config) -> pd.DataFrame:
    start, end = stock_start_end(cfg)
    last_err = None
    for _ in range(cfg.retries + 1):
        try:
            time.sleep(random.uniform(*cfg.sleep_range))
            df = call_with_supported_params(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            if isinstance(df, pd.DataFrame) and len(df) >= 80:
                return normalize_stock_hist(df)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"个股历史拉取失败：{code} last_err={last_err}")


# =========================
# 概念排名
# =========================
def calc_concept_features(hist: pd.DataFrame) -> dict:
    c = pd.to_numeric(hist["close"], errors="coerce")
    v = pd.to_numeric(hist["volume"], errors="coerce")
    a = pd.to_numeric(hist["amount"], errors="coerce")

    ret_5 = c.pct_change(5).iloc[-1]
    ret_20 = c.pct_change(20).iloc[-1]
    accel = ret_5 - ret_20
    amt_ratio = (a.iloc[-1] / (a.rolling(20).mean().iloc[-1] + 1e-12)) - 1.0

    return {
        "ret_5d": float(ret_5),
        "ret_20d": float(ret_20),
        "accel": float(accel),
        "amt_ratio": float(amt_ratio),
        "last_date": hist["date"].iloc[-1],
    }


def build_concept_rank(cfg: Config) -> pd.DataFrame:
    concept_list = fetch_concept_list()
    if cfg.debug_sample and cfg.debug_sample > 0:
        concept_list = concept_list.head(cfg.debug_sample).copy()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows, fail = [], {}

    with ThreadPoolExecutor(max_workers=cfg.max_workers_concept) as ex:
        futs = {
            ex.submit(fetch_concept_hist, r.concept_name, r.concept_code, cfg): (r.concept_code, r.concept_name)
            for r in concept_list.itertuples(index=False)
        }
        for fut in as_completed(futs):
            code, name = futs[fut]
            try:
                hist = fut.result()
                if len(hist) > cfg.concept_lookback:
                    hist = hist.iloc[-cfg.concept_lookback:]
                feats = calc_concept_features(hist)
                feats["concept_code"] = code
                feats["concept_name"] = name
                rows.append(feats)
            except Exception as e:
                msg = str(e)[:150]
                fail[msg] = fail.get(msg, 0) + 1

    if not rows:
        top_errs = sorted(fail.items(), key=lambda x: x[1], reverse=True)[:10]
        raise RuntimeError(f"概念历史全部失败：{top_errs}")

    df = pd.DataFrame(rows)
    df["score"] = (
        0.45 * zscore(df["ret_20d"]) +
        0.35 * zscore(df["accel"]) +
        0.20 * zscore(df["amt_ratio"])
    )
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    last_dt = pd.to_datetime(df["last_date"]).max()
    if (pd.Timestamp.now().normalize() - last_dt.normalize()).days > 14:
        raise RuntimeError(f"⚠️ 数据明显过期：最大 last_date={last_dt.date()}，请检查数据源/限流/网络")

    return df


# =========================
# 波段候选（只输出长周期）
# =========================
def eval_swing_setups(hist: pd.DataFrame, cfg: Config) -> list[dict]:
    """
    返回可能的波段形态信号（可能0/1/2个）：
    1) 回踩MA20
    2) 突破20日新高
    """
    hist = hist.dropna(subset=["open", "high", "low", "close", "volume"])
    if len(hist) < 80:
        return []

    close = hist["close"]
    vol = hist["volume"]

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    vol_ma20 = vol.rolling(20).mean()
    atr14 = compute_atr14(hist)

    last = hist.iloc[-1]
    c = float(last["close"])
    h = float(last["high"])
    l = float(last["low"])

    ma20_last = float(ma20.iloc[-1])
    ma60_last = float(ma60.iloc[-1])
    vol_last = float(vol.iloc[-1])
    vol_ma20_last = float(vol_ma20.iloc[-1]) if not np.isnan(vol_ma20.iloc[-1]) else np.nan
    atr_last = float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else np.nan

    # 趋势前置：波段只做“趋势不弱”
    trend_ok = (c > ma60_last) and (ma20_last > ma60_last)

    if not trend_ok:
        return []

    # 缩量条件（可选）
    vol_ok = True
    if cfg.require_vol_contract:
        if np.isnan(vol_ma20_last):
            vol_ok = False
        else:
            vol_ok = vol_last <= cfg.vol_contract * vol_ma20_last

    setups = []

    # 形态1：回踩MA20（更不容易买在高点）
    pb20 = abs(c - ma20_last) / (ma20_last + 1e-12)
    if pb20 <= cfg.pullback_tol and vol_ok:
        trigger = h * (1.0 + cfg.trigger_buffer)
        # 止损：三者取更紧（更保守）那个
        s1 = l * (1.0 - cfg.stop_low_buffer)
        s2 = ma60_last * (1.0 - cfg.stop_ma60_buffer)
        s3 = (ma20_last - cfg.atr_mult * atr_last) if not np.isnan(atr_last) else s2
        stop = min(s1, s2, s3)
        setups.append({
            "setup": "回踩MA20",
            "trigger_next": trigger,
            "stop_loss": stop,
            "pb20": pb20,
        })

    # 形态2：突破20日新高（趋势确认跟随）
    # 用前一日的20日最高，避免“今天自己算进去”
    high20_prev = hist["high"].rolling(20).max().shift(1).iloc[-1]
    if pd.notna(high20_prev) and c >= float(high20_prev) * 0.995 and vol_ok:
        trigger = float(high20_prev) * (1.0 + cfg.trigger_buffer)
        s1 = l * (1.0 - cfg.stop_low_buffer)
        s2 = ma60_last * (1.0 - cfg.stop_ma60_buffer)
        s3 = (ma20_last - cfg.atr_mult * atr_last) if not np.isnan(atr_last) else s2
        stop = min(s1, s2, s3)
        setups.append({
            "setup": "突破20日新高",
            "trigger_next": trigger,
            "stop_loss": stop,
            "pb20": pb20,
        })

    return setups


def build_swing_candidates(concept_rank: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    输出：
    - SwingCandidates：波段可交易股票
    - Diagnostics：每层过滤的计数，方便你知道为什么会空
    """
    spot = fetch_a_spot()
    top = concept_rank.head(cfg.top_k_concepts).copy()

    diag = {
        "concepts_in": len(top),
        "stocks_after_amount_filter": 0,
        "stocks_hist_fetched": 0,
        "stocks_pass_trend": 0,
        "signals_generated": 0,
    }

    rows = []

    for r in top.itertuples(index=False):
        try:
            cons = fetch_concept_cons(r.concept_name, r.concept_code, cfg)
            cons_code = _pick_col(cons, ["代码", "证券代码"])
            cons_name = _pick_col(cons, ["名称", "证券名称"])
            cons = cons[[cons_code, cons_name]].copy()
            cons.columns = ["code", "stock_name"]
            cons["code"] = cons["code"].astype(str).str.zfill(6)

            m = cons.merge(spot, on="code", how="left")
            m = m.dropna(subset=["amount"])
            m = m[m["amount"] >= cfg.min_amount].copy()
            m = m.sort_values("amount", ascending=False).head(cfg.preselect_top_by_amount)

            diag["stocks_after_amount_filter"] += len(m)

            # 拉历史并评估形态
            for _, rr in m.iterrows():
                code = rr["code"]
                try:
                    hist = fetch_stock_hist(code, cfg)
                    diag["stocks_hist_fetched"] += 1
                except Exception:
                    continue

                setups = eval_swing_setups(hist, cfg)
                if not setups:
                    continue

                # 一只票可能同时满足两种形态，全部输出
                for s in setups:
                    diag["signals_generated"] += 1
                    rows.append({
                        "concept_name": r.concept_name,
                        "concept_score": float(r.score),
                        "ret_20d_concept": float(r.ret_20d),
                        "accel_concept": float(r.accel),
                        "amt_ratio_concept": float(r.amt_ratio),
                        "stock_code": code,
                        "stock_name": rr.get("stock_name", ""),
                        "amount": float(rr["amount"]),
                        "setup": s["setup"],
                        "trigger_next": float(s["trigger_next"]),
                        "stop_loss": float(s["stop_loss"]),
                        "pb20": float(s["pb20"]),
                    })

        except Exception:
            continue

    df = pd.DataFrame(rows)
    diag_df = pd.DataFrame([diag])

    if df.empty:
        return df, diag_df

    # 排序：题材更强 + 流动性更好 + 更贴近MA20（pb20更小）
    df["rank_score"] = (
        0.55 * zscore(df["concept_score"]) +
        0.30 * zscore(df["amount"]) +
        0.15 * (1 - zscore(df["pb20"].clip(lower=0, upper=0.2)))
    )
    df = df.sort_values("rank_score", ascending=False)

    # 每个题材保留前N只
    df = (df.groupby("concept_name", as_index=False)
            .head(cfg.swing_per_concept)
            .reset_index(drop=True))

    return df, diag_df


# =========================
# 主程序
# =========================
def main():
    cfg = Config()
    print("akshare version:", getattr(ak, "__version__", "unknown"))
    print("signature stock_board_concept_hist_em:", inspect.signature(ak.stock_board_concept_hist_em))
    print("✅ 概念/个股历史都显式传 start_date/end_date，不再受默认 20221128 影响。")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ymd = now_ymd()

    concept_rank = build_concept_rank(cfg)
    swing_df, diag_df = build_swing_candidates(concept_rank, cfg)

    xlsx_path = out_dir / f"swing_only_{ymd}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        concept_rank.head(cfg.top_k_concepts).to_excel(w, index=False, sheet_name="ConceptRank")
        swing_df.to_excel(w, index=False, sheet_name="SwingCandidates")
        diag_df.to_excel(w, index=False, sheet_name="Diagnostics")

    print("✅ 输出完成：", xlsx_path)
    print("SwingCandidates:", 0 if swing_df.empty else len(swing_df))
    print("Diagnostics:\n", diag_df.to_string(index=False))


if __name__ == "__main__":
    main()
