# hot_theme_scanner.py
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
# 配置
# =========================
@dataclass
class Config:
    years: int = 5
    lookback: int = 260 * 5

    top_k_concepts: int = 25
    leader_k: int = 12
    max_workers: int = 4
    retries: int = 2
    sleep_range: tuple[float, float] = (0.05, 0.20)

    # 超短候选
    short_candidate_per_concept: int = 2
    short_pct_min: float = 2.0
    short_pct_max: float = 9.3
    min_amount: float = 2e8

    trigger_buffer_short: float = 0.003
    stop_buffer_short: float = 0.005

    # 波段候选（回踩策略）
    swing_candidate_per_concept: int = 2
    pullback_tol: float = 0.02          # 回踩到MA10/20的容忍度 2%
    vol_contract: float = 0.8           # 缩量阈值：< 0.8 * vol_MA20
    trigger_buffer_swing: float = 0.002 # 明日突破今日高点 0.2%
    stop_buffer_swing: float = 0.005    # 今日低点下方 0.5%
    ma20_stop_buffer: float = 0.01      # MA20 下方 1%

    out_dir: str = "output"
    debug_sample: int = 0               # >0 只跑前N个概念调试


# =========================
# 工具函数
# =========================
def now_ymd() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d")


def start_end_dates(cfg: Config) -> tuple[str, str]:
    end = pd.Timestamp.now().strftime("%Y%m%d")
    start = (pd.Timestamp.now() - pd.Timedelta(days=365 * cfg.years)).strftime("%Y%m%d")
    return start, end


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


def normalize_hist(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_col(df, ["日期", "date"])

    close_col = None
    for keys in (["收盘", "close"], ["收盘价"], ["最新价"]):
        try:
            close_col = _pick_col(df, keys)
            break
        except Exception:
            pass
    if close_col is None:
        raise KeyError("找不到收盘列")

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
        raise KeyError("找不到成交量/成交额列")
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
    """个股日线：统一成 date, open, high, low, close, volume, amount"""
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
        raise KeyError("个股历史找不到成交量/成交额列")
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


# =========================
# 数据拉取
# =========================
def fetch_concept_list(cfg: Config) -> pd.DataFrame:
    code_keys = ["板块代码", "代码", "symbol", "code"]
    name_keys = ["板块名称", "名称", "name"]
    last_err = None
    for i in range(cfg.retries + 1):
        try:
            time.sleep(random.uniform(*cfg.sleep_range))
            df = ak.stock_board_concept_name_em()
            code_col = _pick_col(df, code_keys)
            name_col = _pick_col(df, name_keys)
            out = df[[code_col, name_col]].rename(columns={code_col: "concept_code", name_col: "concept_name"})
            out["concept_code"] = out["concept_code"].astype(str)
            out["concept_name"] = out["concept_name"].astype(str)
            return out
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (i + 1))

    ths_api = getattr(ak, "stock_board_concept_name_ths", None)
    if ths_api is not None:
        for i in range(cfg.retries + 1):
            try:
                time.sleep(random.uniform(*cfg.sleep_range))
                df = ths_api()
                code_col = _pick_col(df, code_keys)
                name_col = _pick_col(df, name_keys)
                out = df[[code_col, name_col]].rename(columns={code_col: "concept_code", name_col: "concept_name"})
                out["concept_code"] = out["concept_code"].astype(str)
                out["concept_name"] = out["concept_name"].astype(str)
                return out
            except Exception as e:
                last_err = e
                time.sleep(0.4 * (i + 1))

    out_dir = Path(cfg.out_dir)
    if out_dir.exists():
        cached = sorted(out_dir.glob("concept_rank_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cached:
            df = pd.read_csv(cached[0], encoding="utf-8-sig")
            if {"concept_code", "concept_name"}.issubset(df.columns):
                out = df[["concept_code", "concept_name"]].copy()
                out["concept_code"] = out["concept_code"].astype(str)
                out["concept_name"] = out["concept_name"].astype(str)
                print(f"⚠️ 使用本地概念列表缓存：{cached[0]}")
                return out

    raise RuntimeError(f"概念列表拉取失败（已重试与回退），last_err={last_err}")


def fetch_concept_hist(concept_name: str, concept_code: str, cfg: Config) -> pd.DataFrame:
    start, end = start_end_dates(cfg)
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
                    return normalize_hist(df)
            except Exception as e:
                last_err = e

    for _ in range(cfg.retries + 1):
        try:
            time.sleep(random.uniform(*cfg.sleep_range))
            df = call_with_supported_params(
                ak.stock_board_concept_hist_ths,
                symbol=concept_name, start_date=start, end_date=end
            )
            if isinstance(df, pd.DataFrame) and len(df) >= 60:
                return normalize_hist(df)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"概念历史拉取失败：{concept_name}/{concept_code}，last_err={last_err}")


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
    raise RuntimeError(f"概念成分拉取失败：{concept_name}/{concept_code}，last_err={last_err}")


def fetch_a_spot() -> pd.DataFrame:
    spot = ak.stock_zh_a_spot_em()
    code_col = _pick_col(spot, ["代码", "证券代码"])
    name_col = _pick_col(spot, ["名称", "证券名称"])
    pct_col = _pick_col(spot, ["涨跌幅"])
    amt_col = _pick_col(spot, ["成交额"])

    high_col = None
    low_col = None
    open_col = None
    last_col = None
    for k in ["最高", "最高价"]:
        try:
            high_col = _pick_col(spot, [k]); break
        except Exception:
            pass
    for k in ["最低", "最低价"]:
        try:
            low_col = _pick_col(spot, [k]); break
        except Exception:
            pass
    for k in ["今开", "开盘"]:
        try:
            open_col = _pick_col(spot, [k]); break
        except Exception:
            pass
    for k in ["最新价", "现价", "最新"]:
        try:
            last_col = _pick_col(spot, [k]); break
        except Exception:
            pass

    cols = [code_col, name_col, pct_col, amt_col]
    rename = {code_col: "code", name_col: "name", pct_col: "pct_chg", amt_col: "amount"}
    if high_col: cols.append(high_col); rename[high_col] = "high"
    if low_col: cols.append(low_col); rename[low_col] = "low"
    if open_col: cols.append(open_col); rename[open_col] = "open"
    if last_col: cols.append(last_col); rename[last_col] = "last"

    spot = spot[cols].copy().rename(columns=rename)
    spot["code"] = spot["code"].astype(str).str.zfill(6)
    for c in ["pct_chg", "amount", "high", "low", "open", "last"]:
        if c in spot.columns:
            spot[c] = pd.to_numeric(spot[c], errors="coerce")
    return spot


def fetch_stock_daily_hist(code: str, cfg: Config) -> pd.DataFrame:
    """个股日线（用于MA/回踩/缩量判断）"""
    start, end = start_end_dates(cfg)
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
        except Exception:
            continue
    raise RuntimeError(f"个股历史拉取失败：{code}")


# =========================
# 概念特征/阶段
# =========================
def calc_concept_features(hist: pd.DataFrame) -> dict:
    c = pd.to_numeric(hist["close"], errors="coerce")
    v = pd.to_numeric(hist["volume"], errors="coerce")
    a = pd.to_numeric(hist["amount"], errors="coerce")

    ret_5 = c.pct_change(5).iloc[-1]
    ret_20 = c.pct_change(20).iloc[-1]
    accel = ret_5 - ret_20
    vol_ratio = (v.iloc[-1] / (v.rolling(20).mean().iloc[-1] + 1e-12)) - 1.0
    amt_ratio = (a.iloc[-1] / (a.rolling(20).mean().iloc[-1] + 1e-12)) - 1.0
    high_60 = c.rolling(60).max().iloc[-1]
    breakout = 1.0 if (c.iloc[-1] >= high_60 * 0.995) else 0.0

    return {
        "ret_5d": float(ret_5),
        "ret_20d": float(ret_20),
        "accel": float(accel),
        "vol_ratio": float(vol_ratio),
        "amt_ratio": float(amt_ratio),
        "breakout_60d": float(breakout),
        "last_close": float(c.iloc[-1]),
        "last_date": hist["date"].iloc[-1],
    }


def concept_stage(ret_5d: float, accel: float, amt_ratio: float, breakout_60d: float) -> str:
    r5 = ret_5d * 100
    if amt_ratio <= 0 and accel <= 0:
        return "降温/不看"
    if r5 < 2 and amt_ratio > 0.2 and accel > 0:
        return "孕育"
    if 2 <= r5 <= 10 and accel > 0 and amt_ratio > 0.1:
        return "启动"
    if r5 > 10 and accel > 0 and amt_ratio > 0.2 and breakout_60d >= 0.5:
        return "高潮/谨慎"
    return "扩散"


def build_concept_rank(cfg: Config) -> pd.DataFrame:
    concept_list = fetch_concept_list(cfg)
    if cfg.debug_sample and cfg.debug_sample > 0:
        concept_list = concept_list.head(cfg.debug_sample).copy()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows, fail = [], {}

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = {ex.submit(fetch_concept_hist, r.concept_name, r.concept_code, cfg): (r.concept_code, r.concept_name)
                for r in concept_list.itertuples(index=False)}
        for fut in as_completed(futs):
            code, name = futs[fut]
            try:
                hist = fut.result()
                if len(hist) > cfg.lookback:
                    hist = hist.iloc[-cfg.lookback:]
                feats = calc_concept_features(hist)
                feats["concept_code"] = code
                feats["concept_name"] = name
                rows.append(feats)
            except Exception as e:
                msg = str(e)[:160]
                fail[msg] = fail.get(msg, 0) + 1

    if not rows:
        top_errs = sorted(fail.items(), key=lambda x: x[1], reverse=True)[:10]
        raise RuntimeError(f"没有拉到任何概念历史数据。失败原因Top：{top_errs}")

    df = pd.DataFrame(rows)
    df["score"] = (
        0.35 * zscore(df["ret_5d"]) +
        0.25 * zscore(df["accel"]) +
        0.20 * zscore(df["amt_ratio"]) +
        0.10 * zscore(df["vol_ratio"]) +
        0.10 * df["breakout_60d"]
    )
    df["stage"] = df.apply(lambda r: concept_stage(r["ret_5d"], r["accel"], r["amt_ratio"], r["breakout_60d"]), axis=1)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    last_dt = pd.to_datetime(df["last_date"]).max()
    if (pd.Timestamp.now().normalize() - last_dt.normalize()).days > 14:
        raise RuntimeError(f"⚠️ 数据明显过期：最大 last_date={last_dt.date()}，请检查数据源/限流/网络")

    return df


# =========================
# 超短候选 + 龙头观察
# =========================
def build_short_candidates(concept_rank: pd.DataFrame, cfg: Config):
    spot = fetch_a_spot()
    top = concept_rank.head(cfg.top_k_concepts).copy()

    short_rows, observe_rows = [], []

    for r in top.itertuples(index=False):
        try:
            cons = fetch_concept_cons(r.concept_name, r.concept_code, cfg)
            cons_code = _pick_col(cons, ["代码", "证券代码"])
            cons_name = _pick_col(cons, ["名称", "证券名称"])
            cons = cons[[cons_code, cons_name]].copy()
            cons.columns = ["code", "name"]
            cons["code"] = cons["code"].astype(str).str.zfill(6)

            m = cons.merge(spot, on="code", how="left")
            m = m.dropna(subset=["pct_chg", "amount"])
            m["strength"] = 0.6 * zscore(m["pct_chg"]) + 0.4 * zscore(m["amount"])
            m = m.sort_values("strength", ascending=False).head(cfg.leader_k)

            for _, rr in m.iterrows():
                pct = float(rr["pct_chg"])
                amt = float(rr["amount"])

                base = {
                    "concept_name": r.concept_name,
                    "stage": r.stage,
                    "concept_score": float(r.score),
                    "stock_code": rr["code"],
                    "stock_name": rr["name"],
                    "pct_chg": pct,
                    "amount": amt,
                    "high": rr.get("high", np.nan),
                    "low": rr.get("low", np.nan),
                    "open": rr.get("open", np.nan),
                    "last": rr.get("last", np.nan),
                }

                if pct >= 9.5 or r.stage == "高潮/谨慎":
                    observe_rows.append({**base, "note": "龙头观察：别无脑追，等分歧转强/回踩承接"})
                    continue

                if (cfg.short_pct_min <= pct <= cfg.short_pct_max) and (amt >= cfg.min_amount) and (r.stage in ["启动", "扩散", "孕育"]):
                    high = rr.get("high", np.nan)
                    low = rr.get("low", np.nan)
                    trigger = np.nan
                    stop = np.nan
                    if pd.notna(high) and pd.notna(low):
                        trigger = float(high) * (1.0 + cfg.trigger_buffer_short)
                        stop = float(low) * (1.0 - cfg.stop_buffer_short)

                    short_rows.append({
                        **base,
                        "plan": "次日突破买",
                        "trigger_next": trigger,
                        "stop_loss": stop,
                        "risk_hint": "触发才买；跌破止损就走；别盘中追高",
                    })

        except Exception:
            continue

    short_df = pd.DataFrame(short_rows)
    obs_df = pd.DataFrame(observe_rows)

    if not short_df.empty:
        short_df = (short_df
                    .sort_values(["concept_score", "amount"], ascending=[False, False])
                    .groupby("concept_name", as_index=False)
                    .head(cfg.short_candidate_per_concept)
                    .reset_index(drop=True))

    if not obs_df.empty:
        obs_df = obs_df.sort_values(["concept_score", "amount"], ascending=[False, False]).reset_index(drop=True)

    return short_df, obs_df


# =========================
# 波段候选：趋势回踩买（2～8周）
# =========================
def build_swing_candidates(concept_rank: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    spot = fetch_a_spot()
    top = concept_rank.head(cfg.top_k_concepts).copy()

    # 题材层：更偏“主线”筛选（避免纯情绪）
    top = top[(top["ret_20d"] > 0) & (top["accel"] >= 0) & (top["amt_ratio"] > 0) & (top["stage"] != "降温/不看")].copy()
    top = top.sort_values("score", ascending=False).head(cfg.top_k_concepts)

    swing_rows = []

    for r in top.itertuples(index=False):
        try:
            cons = fetch_concept_cons(r.concept_name, r.concept_code, cfg)
            cons_code = _pick_col(cons, ["代码", "证券代码"])
            cons_name = _pick_col(cons, ["名称", "证券名称"])
            cons = cons[[cons_code, cons_name]].copy()
            cons.columns = ["code", "name"]
            cons["code"] = cons["code"].astype(str).str.zfill(6)

            m = cons.merge(spot, on="code", how="left")
            m = m.dropna(subset=["pct_chg", "amount"])
            # 波段也要流动性，避免小票坑人
            m = m[m["amount"] >= cfg.min_amount].copy()

            # 先用“成交额+强度”挑出少量候选再拉历史，避免太慢
            m["pre_score"] = 0.5 * zscore(m["amount"]) + 0.5 * zscore(m["pct_chg"])
            m = m.sort_values("pre_score", ascending=False).head(cfg.leader_k)

            for _, rr in m.iterrows():
                code = rr["code"]
                try:
                    hist = fetch_stock_daily_hist(code, cfg)
                except Exception:
                    continue

                # 计算MA
                hist = hist.dropna(subset=["close", "volume", "open", "high", "low"])
                if len(hist) < 80:
                    continue

                close = hist["close"]
                vol = hist["volume"]

                ma10 = close.rolling(10).mean()
                ma20 = close.rolling(20).mean()
                ma60 = close.rolling(60).mean()
                vol_ma20 = vol.rolling(20).mean()

                last = hist.iloc[-1]
                c_last = float(last["close"])
                o_last = float(last["open"])
                h_last = float(last["high"])
                l_last = float(last["low"])

                ma10_last = float(ma10.iloc[-1])
                ma20_last = float(ma20.iloc[-1])
                ma60_last = float(ma60.iloc[-1])
                vol_last = float(vol.iloc[-1])
                vol_ma20_last = float(vol_ma20.iloc[-1]) if not np.isnan(vol_ma20.iloc[-1]) else np.nan

                # 条件1：趋势不弱
                trend_ok = (c_last > ma60_last) or (ma20_last > ma60_last)

                # 条件2：回踩到MA10或MA20
                pb10 = abs(c_last - ma10_last) / (ma10_last + 1e-12)
                pb20 = abs(c_last - ma20_last) / (ma20_last + 1e-12)
                pullback_ok = (pb10 <= cfg.pullback_tol) or (pb20 <= cfg.pullback_tol)

                # 条件3：缩量
                vol_ok = (not np.isnan(vol_ma20_last)) and (vol_last < cfg.vol_contract * vol_ma20_last)

                # 条件4：止跌转强
                candle_ok = c_last >= o_last

                if trend_ok and pullback_ok and vol_ok and candle_ok:
                    trigger = h_last * (1.0 + cfg.trigger_buffer_swing)
                    stop1 = l_last * (1.0 - cfg.stop_buffer_swing)
                    stop2 = ma20_last * (1.0 - cfg.ma20_stop_buffer)
                    stop = min(stop1, stop2)

                    swing_rows.append({
                        "concept_name": r.concept_name,
                        "stage": r.stage,
                        "concept_score": float(r.score),
                        "stock_code": code,
                        "stock_name": rr.get("name", ""),
                        "pct_chg": float(rr["pct_chg"]),
                        "amount": float(rr["amount"]),
                        "close": c_last,
                        "MA10": ma10_last,
                        "MA20": ma20_last,
                        "MA60": ma60_last,
                        "pb10": pb10,
                        "pb20": pb20,
                        "vol_last": vol_last,
                        "vol_ma20": vol_ma20_last,
                        "plan": "趋势回踩买（波段）",
                        "trigger_next": trigger,
                        "stop_loss": stop,
                        "why": "趋势OK+回踩MA10/20+缩量+收红",
                    })

        except Exception:
            continue

    swing_df = pd.DataFrame(swing_rows)
    if swing_df.empty:
        return swing_df

    # 每题材保留前N只
    swing_df = (swing_df
                .sort_values(["concept_score", "amount"], ascending=[False, False])
                .groupby("concept_name", as_index=False)
                .head(cfg.swing_candidate_per_concept)
                .reset_index(drop=True))
    return swing_df


# =========================
# 主程序
# =========================
def main():
    cfg = Config()

    print("akshare version:", getattr(ak, "__version__", "unknown"))
    print("signature stock_board_concept_hist_em:", inspect.signature(ak.stock_board_concept_hist_em))
    print("✅ 已显式传 start_date/end_date，避免默认 end_date=20221128。")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ymd = now_ymd()

    concept_rank = build_concept_rank(cfg)
    short_df, obs_df = build_short_candidates(concept_rank, cfg)
    swing_df = build_swing_candidates(concept_rank, cfg)

    xlsx_path = out_dir / f"hot_theme_short_and_swing_{ymd}.xlsx"
    csv_path = out_dir / f"concept_rank_{ymd}.csv"
    concept_rank.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        concept_rank.head(cfg.top_k_concepts).to_excel(w, index=False, sheet_name="ConceptRank")
        short_df.to_excel(w, index=False, sheet_name="ShortCandidates")
        swing_df.to_excel(w, index=False, sheet_name="SwingCandidates")
        obs_df.to_excel(w, index=False, sheet_name="ObserveLeaders")

    print("✅ 输出完成：")
    print("-", csv_path)
    print("-", xlsx_path)
    print("ShortCandidates:", 0 if short_df.empty else len(short_df))
    print("SwingCandidates:", 0 if swing_df.empty else len(swing_df))
    print("ObserveLeaders:", 0 if obs_df.empty else len(obs_df))


if __name__ == "__main__":
    main()
