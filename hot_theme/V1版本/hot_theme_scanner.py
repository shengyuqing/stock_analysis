# hot_theme_scanner.py
# 依赖：pip install akshare pandas numpy openpyxl

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
    # 历史区间：建议 >= 3年（算20/60日指标足够）
    years: int = 5

    # 取回的最大历史长度（交易日）
    lookback: int = 260 * 5

    # 输出前N个概念
    top_k_concepts: int = 20

    # 每个概念挑选N只龙头候选
    leader_k: int = 6

    # 并发数别太高，避免被限流
    max_workers: int = 4

    # 每个接口失败重试次数
    retries: int = 2

    # 随机sleep，降低限流概率
    sleep_range: tuple[float, float] = (0.05, 0.20)

    # 输出目录
    out_dir: str = "output"

    # 调试：>0 只跑前N个概念（比如 50），跑通后改回0
    debug_sample: int = 0


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
    """
    根据函数签名过滤参数：避免不同 akshare 版本参数不一致导致 TypeError
    """
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return func(**filtered)


def normalize_hist(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一历史行情字段：date, close, volume, amount
    兼容 EM/THS 不同列名
    """
    date_col = _pick_col(df, ["日期", "date"])

    # 收盘列可能叫：收盘 / 收盘价 / 最新价
    close_col = None
    for keys in (["收盘", "close"], ["收盘价"], ["最新价"]):
        try:
            close_col = _pick_col(df, keys)
            break
        except Exception:
            pass
    if close_col is None:
        raise KeyError("找不到收盘列")

    # 成交量/成交额可能缺一个
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
    """
    修复重点：显式传 start_date/end_date，避免默认 end_date='20221128'
    先尝试 EM（东方财富），失败再 fallback THS（同花顺）
    """
    start, end = start_end_dates(cfg)
    last_err = None

    # 1) 东方财富：优先用名称（很多版本 symbol=名称），再试 code
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
                if isinstance(df, pd.DataFrame) and len(df) >= 40:
                    return normalize_hist(df)
            except Exception as e:
                last_err = e

    # 2) fallback：同花顺概念指数历史（一般也是 symbol=名称）
    for _ in range(cfg.retries + 1):
        try:
            time.sleep(random.uniform(*cfg.sleep_range))
            df = call_with_supported_params(
                ak.stock_board_concept_hist_ths,
                symbol=concept_name,
                start_date=start,
                end_date=end
            )
            if isinstance(df, pd.DataFrame) and len(df) >= 40:
                return normalize_hist(df)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"概念历史拉取失败：{concept_name}/{concept_code}，last_err={last_err}")


def fetch_concept_cons(concept_name: str, concept_code: str, cfg: Config) -> pd.DataFrame:
    """
    概念成分：优先名称，再试 code
    """
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


# =========================
# 特征与评分
# =========================
def calc_features(hist: pd.DataFrame) -> dict:
    if len(hist) < 60:
        return {}

    c = pd.to_numeric(hist["close"], errors="coerce")
    v = pd.to_numeric(hist["volume"], errors="coerce")
    a = pd.to_numeric(hist["amount"], errors="coerce")

    # 5日、20日收益 + “加速度”
    ret_5 = c.pct_change(5).iloc[-1]
    ret_20 = c.pct_change(20).iloc[-1]
    accel = ret_5 - ret_20

    # 量能相对20日均值放大
    vol_ratio = (v.iloc[-1] / (v.rolling(20).mean().iloc[-1] + 1e-12)) - 1.0
    amt_ratio = (a.iloc[-1] / (a.rolling(20).mean().iloc[-1] + 1e-12)) - 1.0

    # 60日新高附近（避免“反弹但没突破”）
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
                feats = calc_features(hist)
                if feats:
                    feats["concept_code"] = code
                    feats["concept_name"] = name
                    rows.append(feats)
            except Exception as e:
                msg = str(e)[:140]
                fail[msg] = fail.get(msg, 0) + 1

    if not rows:
        top_errs = sorted(fail.items(), key=lambda x: x[1], reverse=True)[:10]
        raise RuntimeError(f"没有拉到任何概念历史数据。失败原因Top：{top_errs}")

    df = pd.DataFrame(rows)

    # ✅ 热度评分：短期强度 + 加速度 + 量能 + 突破
    df["score"] = (
        0.35 * zscore(df["ret_5d"]) +
        0.25 * zscore(df["accel"]) +
        0.20 * zscore(df["amt_ratio"]) +
        0.10 * zscore(df["vol_ratio"]) +
        0.10 * df["breakout_60d"]
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # ✅ 防止你再遇到“日期停在很久以前”
    last_dt = pd.to_datetime(df["last_date"]).max()
    if (pd.Timestamp.now().normalize() - last_dt.normalize()).days > 14:
        raise RuntimeError(f"⚠️ 数据明显过期：最大 last_date={last_dt.date()}，请检查数据源是否被限流/接口不可用")

    return df


def pick_leaders_for_top_concepts(rank_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    top = rank_df.head(cfg.top_k_concepts).copy()

    # 全A实时
    spot = ak.stock_zh_a_spot_em()
    code_col = _pick_col(spot, ["代码", "证券代码"])
    name_col = _pick_col(spot, ["名称", "证券名称"])
    pct_col = _pick_col(spot, ["涨跌幅"])
    amt_col = _pick_col(spot, ["成交额"])

    spot = spot[[code_col, name_col, pct_col, amt_col]].copy()
    spot.columns = ["code", "name", "pct_chg", "amount"]
    spot["code"] = spot["code"].astype(str).str.zfill(6)
    spot["pct_chg"] = pd.to_numeric(spot["pct_chg"], errors="coerce")
    spot["amount"] = pd.to_numeric(spot["amount"], errors="coerce")

    leader_rows = []
    for r in top.itertuples(index=False):
        try:
            cons = fetch_concept_cons(r.concept_name, r.concept_code, cfg)
            cons_code = _pick_col(cons, ["代码", "证券代码"])
            cons_name = _pick_col(cons, ["名称", "证券名称"])
            cons = cons[[cons_code, cons_name]].copy()
            cons.columns = ["code", "name"]
            cons["code"] = cons["code"].astype(str).str.zfill(6)

            merged = cons.merge(spot[["code", "pct_chg", "amount"]], on="code", how="left")
            merged["leader_score"] = 0.6 * zscore(merged["pct_chg"]) + 0.4 * zscore(merged["amount"])
            merged = merged.sort_values("leader_score", ascending=False).head(cfg.leader_k)

            for _, rr in merged.iterrows():
                leader_rows.append({
                    "concept_code": r.concept_code,
                    "concept_name": r.concept_name,
                    "concept_score": float(r.score),
                    "stock_code": rr["code"],
                    "stock_name": rr["name"],
                    "pct_chg": rr.get("pct_chg", np.nan),
                    "amount": rr.get("amount", np.nan),
                })
        except Exception:
            continue

    return pd.DataFrame(leader_rows)


# =========================
# 主程序
# =========================
def main():
    cfg = Config()
    print("akshare version:", getattr(ak, "__version__", "unknown"))
    print("hist signature:", inspect.signature(ak.stock_board_concept_hist_em))
    print("将显式传入 start_date/end_date，避免默认 end_date=20221128 的问题。")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ymd = now_ymd()

    rank_df = build_concept_rank(cfg)
    rank_path = out_dir / f"concept_rank_{ymd}.csv"
    rank_df.to_csv(rank_path, index=False, encoding="utf-8-sig")

    leaders_df = pick_leaders_for_top_concepts(rank_df, cfg)
    xlsx_path = out_dir / f"top_concepts_leaders_{ymd}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        rank_df.head(cfg.top_k_concepts).to_excel(w, index=False, sheet_name="TopConcepts")
        leaders_df.to_excel(w, index=False, sheet_name="Leaders")

    print("✅ 已输出：")
    print(f"- {rank_path}")
    print(f"- {xlsx_path}")
    print("✅ last_date示例：", rank_df["last_date"].iloc[0])


if __name__ == "__main__":
    main()
