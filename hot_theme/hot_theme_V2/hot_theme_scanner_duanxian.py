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
    years: int = 5                 # 历史区间（防止默认 end_date=20221128）
    lookback: int = 260 * 5

    top_k_concepts: int = 25       # 题材榜输出数量
    leader_k: int = 10             # 每个题材抓多少只强势股用于筛选
    max_workers: int = 4
    retries: int = 2
    sleep_range: tuple[float, float] = (0.05, 0.20)

    # 超短：交易候选筛选
    candidate_per_concept: int = 2     # 每个题材最终给几只可交易候选
    pct_chg_min: float = 2.0           # 候选最低涨幅（太弱没意义）
    pct_chg_max: float = 9.3           # 候选最高涨幅（接近涨停的先不当“突破买”）
    min_amount: float = 2e8            # 最低成交额过滤（2亿，按你口味调）

    # 触发价/止损缓冲（避免刚好碰一下就成交/止损）
    trigger_buffer: float = 0.003      # 明日突破买：今天最高价 * (1+0.3%)
    stop_buffer: float = 0.005         # 止损：今天最低价 * (1-0.5%)

    out_dir: str = "output"
    debug_sample: int = 0              # >0 只跑前N个概念调试（例如50）


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
    """不同 akshare 版本参数可能不同，按签名过滤参数，避免 TypeError。"""
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


def fetch_concept_hist(concept_name: str, concept_code: str, cfg: Config) -> pd.DataFrame:
    """显式 start_date/end_date，彻底绕开默认 end_date=20221128 的坑。"""
    start, end = start_end_dates(cfg)
    last_err = None

    # 1) 东方财富：优先概念名称，再试代码
    for sym in [concept_name, concept_code]:
        for _ in range(cfg.retries + 1):
            try:
                time.sleep(random.uniform(*cfg.sleep_range))
                df = call_with_supported_params(
                    ak.stock_board_concept_hist_em,
                    symbol=sym,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust=""
                )
                if isinstance(df, pd.DataFrame) and len(df) >= 60:
                    return normalize_hist(df)
            except Exception as e:
                last_err = e

    # 2) fallback：同花顺概念指数历史（一般 symbol=名称）
    for _ in range(cfg.retries + 1):
        try:
            time.sleep(random.uniform(*cfg.sleep_range))
            df = call_with_supported_params(
                ak.stock_board_concept_hist_ths,
                symbol=concept_name,
                start_date=start,
                end_date=end
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
    """全A实时快照：用于拿 今日最高/最低/开盘/最新/成交额/涨跌幅"""
    spot = ak.stock_zh_a_spot_em()

    code_col = _pick_col(spot, ["代码", "证券代码"])
    name_col = _pick_col(spot, ["名称", "证券名称"])
    pct_col = _pick_col(spot, ["涨跌幅"])
    amt_col = _pick_col(spot, ["成交额"])

    # 这些列大多数版本都有：最高/最低/今开/最新价（如缺会在下面兜底）
    high_col = None
    low_col = None
    open_col = None
    last_col = None
    for k in ["最高", "最高价"]:
        try:
            high_col = _pick_col(spot, [k])
            break
        except Exception:
            pass
    for k in ["最低", "最低价"]:
        try:
            low_col = _pick_col(spot, [k])
            break
        except Exception:
            pass
    for k in ["今开", "开盘"]:
        try:
            open_col = _pick_col(spot, [k])
            break
        except Exception:
            pass
    for k in ["最新价", "现价", "最新"]:
        try:
            last_col = _pick_col(spot, [k])
            break
        except Exception:
            pass

    cols = [code_col, name_col, pct_col, amt_col]
    rename = {code_col: "code", name_col: "name", pct_col: "pct_chg", amt_col: "amount"}
    if high_col:
        cols.append(high_col); rename[high_col] = "high"
    if low_col:
        cols.append(low_col); rename[low_col] = "low"
    if open_col:
        cols.append(open_col); rename[open_col] = "open"
    if last_col:
        cols.append(last_col); rename[last_col] = "last"

    spot = spot[cols].copy().rename(columns=rename)
    spot["code"] = spot["code"].astype(str).str.zfill(6)
    for c in ["pct_chg", "amount", "high", "low", "open", "last"]:
        if c in spot.columns:
            spot[c] = pd.to_numeric(spot[c], errors="coerce")
    return spot


# =========================
# 指标/评分/阶段
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
    """超短用的粗分层：避免你追在高潮顶上。"""
    # 用 ret_5d 绝对值更直观
    r5 = ret_5d * 100

    if amt_ratio <= 0 and accel <= 0:
        return "降温/不看"
    if r5 < 2 and amt_ratio > 0.2 and accel > 0:
        return "孕育"
    if 2 <= r5 <= 10 and accel > 0 and amt_ratio > 0.1:
        return "启动"
    if r5 > 10 and accel > 0 and amt_ratio > 0.2 and breakout_60d >= 0.5:
        return "高潮/谨慎"
    # 其它情况归到“扩散/跟随”
    return "扩散"


def build_concept_rank(cfg: Config) -> pd.DataFrame:
    concept_list = fetch_concept_list()
    if cfg.debug_sample and cfg.debug_sample > 0:
        concept_list = concept_list.head(cfg.debug_sample).copy()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = []
    fail = {}

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = {
            ex.submit(fetch_concept_hist, r.concept_name, r.concept_code, cfg): (r.concept_code, r.concept_name)
            for r in concept_list.itertuples(index=False)
        }
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

    # 过期保护
    last_dt = pd.to_datetime(df["last_date"]).max()
    if (pd.Timestamp.now().normalize() - last_dt.normalize()).days > 14:
        raise RuntimeError(f"⚠️ 数据明显过期：最大 last_date={last_dt.date()}，请检查数据源/限流/网络")

    return df


# =========================
# 超短：候选股与交易计划
# =========================
def build_short_term_candidates(concept_rank: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回：TradeCandidates（可交易） + ObserveLeaders（观察）
    """
    spot = fetch_a_spot()

    # 只取“启动/扩散”题材做超短候选（高潮只进观察）
    top = concept_rank.head(cfg.top_k_concepts).copy()

    trade_rows = []
    observe_rows = []

    for r in top.itertuples(index=False):
        try:
            cons = fetch_concept_cons(r.concept_name, r.concept_code, cfg)
            cons_code = _pick_col(cons, ["代码", "证券代码"])
            cons_name = _pick_col(cons, ["名称", "证券名称"])
            cons = cons[[cons_code, cons_name]].copy()
            cons.columns = ["code", "name"]
            cons["code"] = cons["code"].astype(str).str.zfill(6)

            m = cons.merge(spot, on="code", how="left", suffixes=("", "_spot"))
            # 过滤掉无行情的
            m = m.dropna(subset=["pct_chg", "amount"])

            # 先挑强势：涨幅+成交额（用于龙头/候选筛选的基础池）
            m["strength"] = 0.6 * zscore(m["pct_chg"]) + 0.4 * zscore(m["amount"])
            m = m.sort_values("strength", ascending=False).head(cfg.leader_k)

            for _, rr in m.iterrows():
                pct = float(rr["pct_chg"])
                amt = float(rr["amount"])

                row_base = {
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

                # 观察：涨停/超强（尤其高潮段）
                if pct >= 9.5 or r.stage == "高潮/谨慎":
                    observe_rows.append({**row_base, "note": "龙头观察：别无脑追，等分歧转强/回踩承接"})
                    continue

                # 可交易候选：不接近涨停 + 成交额够大 + 涨幅不太弱
                if (cfg.pct_chg_min <= pct <= cfg.pct_chg_max) and (amt >= cfg.min_amount) and (r.stage in ["启动", "扩散", "孕育"]):
                    high = rr.get("high", np.nan)
                    low = rr.get("low", np.nan)

                    # 明日突破触发价/止损
                    trigger = np.nan
                    stop = np.nan
                    if pd.notna(high) and pd.notna(low):
                        trigger = float(high) * (1.0 + cfg.trigger_buffer)
                        stop = float(low) * (1.0 - cfg.stop_buffer)

                    trade_rows.append({
                        **row_base,
                        "plan": "次日突破买",
                        "trigger_next": trigger,
                        "stop_loss": stop,
                        "risk_hint": "触发才买；跌破止损就走；别盘中追高",
                    })

        except Exception:
            continue

    trade_df = pd.DataFrame(trade_rows)
    obs_df = pd.DataFrame(observe_rows)

    if not trade_df.empty:
        # 每个题材只保留前N个可交易候选（避免你候选池太大乱掉）
        trade_df = (trade_df
                    .sort_values(["concept_score", "amount"], ascending=[False, False])
                    .groupby("concept_name", as_index=False)
                    .head(cfg.candidate_per_concept)
                    .reset_index(drop=True))

    if not obs_df.empty:
        obs_df = obs_df.sort_values(["concept_score", "amount"], ascending=[False, False]).reset_index(drop=True)

    return trade_df, obs_df


# =========================
# 主程序
# =========================
def main():
    cfg = Config()

    print("akshare version:", getattr(ak, "__version__", "unknown"))
    print("signature stock_board_concept_hist_em:", inspect.signature(ak.stock_board_concept_hist_em))
    print("✅ 本脚本已显式传 start_date/end_date，避免默认 end_date=20221128 的坑。")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ymd = now_ymd()

    concept_rank = build_concept_rank(cfg)

    trade_df, obs_df = build_short_term_candidates(concept_rank, cfg)

    # 输出
    xlsx_path = out_dir / f"hot_theme_short_{ymd}.xlsx"
    csv_path = out_dir / f"concept_rank_{ymd}.csv"
    concept_rank.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        concept_rank.head(cfg.top_k_concepts).to_excel(w, index=False, sheet_name="ConceptRank")
        trade_df.to_excel(w, index=False, sheet_name="TradeCandidates")
        obs_df.to_excel(w, index=False, sheet_name="ObserveLeaders")

    print("✅ 输出完成：")
    print("-", csv_path)
    print("-", xlsx_path)
    if not concept_rank.empty:
        print("ConceptRank 示例 last_date:", pd.to_datetime(concept_rank["last_date"].iloc[0]).date())
    print("TradeCandidates 数量:", 0 if trade_df.empty else len(trade_df))
    print("ObserveLeaders 数量:", 0 if obs_df.empty else len(obs_df))


if __name__ == "__main__":
    main()
