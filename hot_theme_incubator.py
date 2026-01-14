# hot_theme_incubator_csvcache.py
# pip install -U akshare pandas numpy tqdm

from __future__ import annotations
import os, time, math, traceback, gzip
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import akshare as ak

import re,inspect

def norm_code(x) -> str | None:
    """把 'SH600000'/'600000.0'/'600000' -> '600000'，匹配不到返回 None"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    m = re.search(r"(\d{6})", s)
    return m.group(1) if m else None


CFG = {
    "out_dir": "./output1",
    "cache_dir": "./cache",

    # CSV gzip 缓存
    "concept_cache_days": 7,

    # 公告
    "notice_recent_pages": 20,
    "notice_lookback_days": 60,

    # 机构调研
    "jgdy_lookback_days": 120,
    "jgdy_focus_days_1": 7,
    "jgdy_focus_days_2": 30,

    # 概念价格/资金惩罚
    "concept_price_days": 90,
    "concept_return_days": 20,
    "concept_fund_days": 20,

    "top_concepts": 25,
    "max_stocks_per_concept": 40,

    # 避免追高（长周期埋伏池）
    "max_60d_return": 0.30,
    "max_close_over_ma60": 1.15,
    "min_turnover": 8e7,
    "exclude_st": True,

    # 调试
    "debug_limit_concepts": 0,   # >0 只拉前N个概念用于调试
}

CFG.update({
    # 概念过滤：去掉指数/通道/规则类
    "concept_member_cnt_min": 15,
    "concept_member_cnt_max": 500,
    "concept_exclude_keywords": [
        "HS", "沪深", "上证", "中证", "深证", "创业板", "科创",
        "MSCI", "标普", "纳斯达克", "道琼斯", "罗素",
        "指数", "成份", "ETF",
        "沪股通", "深股通", "港股通", "融资融券",
        "A股", "B股", "H股", "北证",
        "韩国", "首尔",  # 你榜单里出现“标普首尔”这类明显非A股题材
    ],

    # 公告类型：收敛为更像“产业催化”的
    "notice_report_types": ["重大事项", "融资公告", "资产重组"],
})


# ---------------- utilities ----------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = np.nanmean(s.values)
    sd = np.nanstd(s.values)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def safe_ak_call(fn, *args, retries=3, sleep=1.2, **kwargs):
    last_err = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(sleep * (i + 1))
    raise last_err

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_csv_cache(path: str, max_age_days: int):
    if not os.path.exists(path):
        return None
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    if datetime.now() - mtime > timedelta(days=max_age_days):
        return None
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        # gzip 兼容
        try:
            return pd.read_csv(path, compression="gzip", encoding="utf-8")
        except Exception:
            return None

def save_csv_cache(df: pd.DataFrame, path: str):
    # gzip 压缩，文件小很多
    df.to_csv(path, index=False, encoding="utf-8", compression="gzip")


# ---------------- concept members cache ----------------
def fetch_concept_names() -> list[str]:
    df = safe_ak_call(ak.stock_board_concept_name_em)
    name_col = pick_col(df, ["板块名称", "概念名称", "名称"])
    if not name_col:
        raise RuntimeError(f"概念列表字段未知，实际列：{list(df.columns)}")
    names = df[name_col].astype(str).dropna().unique().tolist()
    names = [x.strip() for x in names if x and x.strip()]
    if CFG["debug_limit_concepts"] and CFG["debug_limit_concepts"] > 0:
        names = names[:CFG["debug_limit_concepts"]]
    return names

def build_concept_members_cache(cache_path: str) -> pd.DataFrame:
    concepts = fetch_concept_names()
    rows = []
    try:
        from tqdm import tqdm
        it = tqdm(concepts, desc="拉取概念成份", ncols=90)
    except Exception:
        it = concepts

    for concept in it:
        try:
            cons = safe_ak_call(ak.stock_board_concept_cons_em, symbol=concept)
            code_col = pick_col(cons, ["代码", "股票代码"])
            name_col = pick_col(cons, ["名称", "股票简称"])
            if not code_col:
                continue
            tmp = pd.DataFrame({
                "概念": concept,
                "代码": cons[code_col].apply(norm_code),
                "股票简称": cons[name_col].astype(str) if name_col else ""
            })
            tmp = tmp.dropna(subset=["代码"])
            rows.append(tmp)
        except Exception:
            continue

    if not rows:
        raise RuntimeError("概念成份全量拉取失败：rows为空。")

    out = pd.concat(rows, ignore_index=True).drop_duplicates()
    save_csv_cache(out, cache_path)
    return out

def get_concept_members() -> pd.DataFrame:
    _ensure_dir(CFG["cache_dir"])
    cache_path = os.path.join(CFG["cache_dir"], "concept_members.csv.gz")
    df = load_csv_cache(cache_path, CFG["concept_cache_days"])
    if df is not None and len(df) > 0:
        return df
    return build_concept_members_cache(cache_path)


# ---------------- leading signals: notices ----------------
def _notice_call_compat(rt: str, pages: str):
    """
    兼容不同 AkShare 版本的 stock_notice_report 调用方式：
    - 可能是 report_type / symbol / type
    - 可能 recent_page 参数名不存在
    - 可能只支持位置参数
    """
    fn = ak.stock_notice_report

    attempts = [
        lambda: fn(report_type=rt, recent_page=pages),
        lambda: fn(symbol=rt, recent_page=pages),
        lambda: fn(type=rt, recent_page=pages),

        # 有些版本 recent_page 参数名不对或没有，试试只传类型
        lambda: fn(report_type=rt),
        lambda: fn(symbol=rt),
        lambda: fn(type=rt),

        # 试位置参数： (rt, pages) / (rt,)
        lambda: fn(rt, pages),
        lambda: fn(rt),

        # 最后兜底：不传（有的版本默认“全部”）
        lambda: fn(),
    ]

    last = None
    for i, f in enumerate(attempts, 1):
        try:
            df = safe_ak_call(f, retries=1, sleep=0.1)  # 单次尝试即可
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                return df
        except Exception as e:
            last = e
            continue

    raise last if last else RuntimeError("stock_notice_report 全部调用方式都失败")


def fetch_notices() -> pd.DataFrame:
    # report_types = ["重大事项", "融资公告", "资产重组", "信息变更", "持股变动", "风险提示"]
    report_types = CFG["notice_report_types"]
    scan_days = CFG["notice_lookback_days"]

    print("[notice] signature:", inspect.signature(ak.stock_notice_report))

    all_parts = []

    for rt in report_types:
        rt_parts = []

        for k in range(scan_days + 1):
            day = (datetime.now() - timedelta(days=k)).strftime("%Y%m%d")
            try:
                df = safe_ak_call(ak.stock_notice_report, symbol=rt, date=day, retries=2, sleep=0.8)
                if df is None or len(df) == 0:
                    continue

                df = df.copy()
                df["report_type"] = rt

                # 字段兼容
                df = df.rename(columns={
                    pick_col(df, ["代码", "股票代码", "证券代码"]): "代码",
                    pick_col(df, ["名称", "股票简称", "证券简称"]): "名称",
                    pick_col(df, ["公告标题", "标题"]): "公告标题",
                    pick_col(df, ["公告类型", "类型"]): "公告类型",
                    pick_col(df, ["公告日期", "日期"]): "公告日期",
                })

                if "代码" not in df.columns or "公告日期" not in df.columns:
                    continue

                df["代码"] = df["代码"].apply(norm_code)
                df["公告日期"] = pd.to_datetime(df["公告日期"], errors="coerce").dt.date
                df = df.dropna(subset=["代码", "公告日期"])

                if len(df) == 0:
                    continue

                # ✅ 诊断：一定要放在 df 定义并清洗之后
                if k < 3:
                    dmin, dmax = df["公告日期"].min(), df["公告日期"].max()
                    print(f"[notice-diag] rt={rt} req_date={day} df_date_range={dmin} ~ {dmax} rows={len(df)}")

                rt_parts.append(df)

            except Exception:
                continue

        # ✅ 去重统计只针对当前 rt
        if rt_parts:
            tmp = pd.concat(rt_parts, ignore_index=True)
            uniq = tmp.drop_duplicates(subset=["代码", "公告标题", "公告日期"])
            print(
                f"[notice] {rt}: raw_sum={len(tmp)} "
                f"unique={len(uniq)} "
                f"date_range={uniq['公告日期'].min()} ~ {uniq['公告日期'].max()}"
            )
            all_parts.append(uniq)  # 直接存去重后的，减少后续量
        else:
            print(f"[notice] {rt}: no data in {scan_days}d")

    if not all_parts:
        return pd.DataFrame(columns=["代码", "名称", "公告标题", "公告类型", "公告日期", "report_type"])

    out = pd.concat(all_parts, ignore_index=True).drop_duplicates()
    dmin, dmax = out["公告日期"].min(), out["公告日期"].max()
    print(f"[notice] merged: {len(out)} rows, date {dmin} ~ {dmax}")
    return out

def stock_notice_features(notice_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["代码", "notice_7d", "notice_30d", "notice_lookback", "notice_accel", "last_notice"]
    if notice_df is None or len(notice_df) == 0:
        return pd.DataFrame(columns=cols)

    today = datetime.now().date()
    lookback = today - timedelta(days=CFG["notice_lookback_days"])
    df = notice_df[notice_df["公告日期"].notna() & (notice_df["公告日期"] >= lookback)].copy()

    if df.empty:
        return pd.DataFrame(columns=cols)

    d7 = today - timedelta(days=7)
    d30 = today - timedelta(days=30)

    df["is7"] = (df["公告日期"] >= d7).astype(int)
    df["is30"] = (df["公告日期"] >= d30).astype(int)

    feat = (df.groupby("代码", as_index=False)
              .agg(
                  notice_7d=("is7", "sum"),
                  notice_30d=("is30", "sum"),
                  notice_lookback=("公告日期", "size"),
                  last_notice=("公告日期", "max"),
              ))

    feat["notice_accel"] = (feat["notice_7d"] + 1) / (feat["notice_30d"] + 3)
    return feat


# ---------------- leading signals: institutional research ----------------
def fetch_jgdy() -> pd.DataFrame:
    start_dt = datetime.now() - timedelta(days=CFG["jgdy_lookback_days"])
    start_str = _yyyymmdd(start_dt)
    df = safe_ak_call(ak.stock_jgdy_tj_em, date=start_str)
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    code_col = pick_col(df, ["代码", "股票代码"])
    name_col = pick_col(df, ["名称", "股票简称"])
    cnt_col = pick_col(df, ["接待机构数量"])
    day_col = pick_col(df, ["接待日期", "调研日期"])

    if not code_col or not cnt_col or not day_col:
        return pd.DataFrame()

    df = df.rename(columns={
        code_col: "代码",
        name_col: "名称" if name_col else "名称",
        cnt_col: "接待机构数量",
        day_col: "调研日期",
    })
    df["代码"] = df["代码"].apply(norm_code)
    df = df.dropna(subset=["代码"])
    df["调研日期"] = pd.to_datetime(df["调研日期"], errors="coerce").dt.date
    df["接待机构数量"] = pd.to_numeric(df["接待机构数量"], errors="coerce").fillna(0).astype(int)
    return df

def stock_jgdy_features(jgdy_df: pd.DataFrame) -> pd.DataFrame:
    if jgdy_df is None or len(jgdy_df) == 0 or "调研日期" not in jgdy_df.columns:
        return pd.DataFrame(columns=["代码", "jgdy_7d", "jgdy_30d", "jgdy_lookback", "jgdy_accel", "last_jgdy"])

    today = datetime.now().date()
    d7 = today - timedelta(days=CFG["jgdy_focus_days_1"])
    d30 = today - timedelta(days=CFG["jgdy_focus_days_2"])
    lookback = today - timedelta(days=CFG["jgdy_lookback_days"])

    df = jgdy_df[jgdy_df["调研日期"].notna() & (jgdy_df["调研日期"] >= lookback)].copy()
    g = df.groupby("代码", as_index=True)

    last_jgdy = g["调研日期"].max()
    jgdy_7d = g.apply(lambda x: x.loc[x["调研日期"] >= d7, "接待机构数量"].sum()).astype(int)
    jgdy_30d = g.apply(lambda x: x.loc[x["调研日期"] >= d30, "接待机构数量"].sum()).astype(int)
    jgdy_lb = g["接待机构数量"].sum().astype(int)
    jgdy_accel = (jgdy_7d + 1) / (jgdy_30d + 5)

    feat = pd.DataFrame({
        "代码": jgdy_lb.index,
        "jgdy_7d": jgdy_7d.values,
        "jgdy_30d": jgdy_30d.values,
        "jgdy_lookback": jgdy_lb.values,
        "jgdy_accel": jgdy_accel.values,
        "last_jgdy": last_jgdy.values,
    })
    return feat


# ---------------- concept penalties (optional) ----------------
def concept_price_return(concept: str) -> float:
    try:
        end = datetime.now()
        start = end - timedelta(days=CFG["concept_price_days"])
        df = safe_ak_call(
            ak.stock_board_concept_hist_em,
            symbol=concept, period="daily",
            start_date=_yyyymmdd(start),
            end_date=_yyyymmdd(end),
            adjust=""
        )
        if df is None or len(df) < (CFG["concept_return_days"] + 2):
            return np.nan
        close_col = pick_col(df, ["收盘", "close"])
        if not close_col:
            return np.nan
        closes = pd.to_numeric(df[close_col], errors="coerce").dropna()
        if len(closes) < (CFG["concept_return_days"] + 2):
            return np.nan
        ret = closes.iloc[-1] / closes.iloc[-(CFG["concept_return_days"] + 1)] - 1
        return float(ret)
    except Exception:
        return np.nan

def concept_fund_20d(concept: str) -> float:
    try:
        df = safe_ak_call(ak.stock_concept_fund_flow_hist, symbol=concept)
        if df is None or len(df) == 0:
            return np.nan
        val_col = pick_col(df, ["主力净流入-净额"])
        if not val_col:
            return np.nan
        s = pd.to_numeric(df[val_col], errors="coerce").dropna()
        if len(s) == 0:
            return np.nan
        return float(s.tail(CFG["concept_fund_days"]).sum())
    except Exception:
        return np.nan


def build_concept_scores(members, notice_feat, jgdy_feat) -> pd.DataFrame:
    m = members.copy()
    m["代码"] = m["代码"].astype(str).str.zfill(6)

    x = m[["概念", "代码", "股票简称"]].drop_duplicates()

    notice_feat = notice_feat if notice_feat is not None else pd.DataFrame(columns=["代码"])
    jgdy_feat = jgdy_feat if jgdy_feat is not None else pd.DataFrame(columns=["代码"])

    x = x.merge(notice_feat, on="代码", how="left").merge(jgdy_feat, on="代码", how="left")
    for c in ["notice_7d", "notice_30d", "notice_lookback", "notice_accel",
              "jgdy_7d", "jgdy_30d", "jgdy_lookback", "jgdy_accel"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    def n_hit(df, col, thr=1):
        return int((df[col] >= thr).sum()) if col in df.columns else 0

    g = x.groupby("概念", as_index=False)
    out = g.apply(lambda df: pd.Series({
        "notice_hit_7d": n_hit(df, "notice_7d", 1),
        "notice_hit_30d": n_hit(df, "notice_30d", 1),
        "notice_accel_mean": float(df["notice_accel"].replace([np.inf, -np.inf], np.nan).fillna(0).mean()),
        "jgdy_sum_30d": float(df["jgdy_30d"].sum()),
        "jgdy_accel_mean": float(df["jgdy_accel"].replace([np.inf, -np.inf], np.nan).fillna(0).mean()),
        "member_cnt": int(len(df)),
    })).reset_index(drop=True)

    # ====== [新增] 过滤非题材概念：宽基/通道/风格/成份 ======
    MAX_MEMBER = 250
    EXCLUDE = ["指数", "成份", "HS", "沪深", "上证", "中证", "深证", "MSCI", "标普", "富时",
               "沪股通", "深股通", "港股通", "融资融券", "百元股", "500", "300", "100", "50"]

    # 注意：你的列名可能是 concept_member_cnt（你表里是 member_cnt / concept_member_cnt 二选一）
    member_col = "concept_member_cnt" if "concept_member_cnt" in out.columns else "member_cnt"

    out = out[(out[member_col] >= 15) & (out[member_col] <= MAX_MEMBER)].copy()
    pat = "|".join(EXCLUDE)
    out = out[~out["概念"].astype(str).str.contains(pat, regex=True, na=False)].copy()

    # --- 过滤：member_cnt 过大/过小 & 关键词黑名单 ---
    out = out[(out["member_cnt"] >= CFG["concept_member_cnt_min"]) &
              (out["member_cnt"] <= CFG["concept_member_cnt_max"])].copy()

    pat = "|".join(map(repr, CFG["concept_exclude_keywords"])).replace("'", "")
    out = out[~out["概念"].astype(str).str.contains(pat, regex=True, na=False)].copy()

    # --- 规模归一化：用“信号密度”替代绝对量 ---
    den = np.sqrt(out["member_cnt"].clip(lower=1))
    out["notice_rate_7d"] = out["notice_hit_7d"] / den
    out["notice_rate_30d"] = out["notice_hit_30d"] / den
    out["jgdy_rate_30d"] = out["jgdy_sum_30d"] / den

    # --- 重新zscore：用 rate + accel，少用绝对量 ---
    out["z_notice_rate_7d"] = _zscore(out["notice_rate_7d"])
    out["z_notice_accel"] = _zscore(out["notice_accel_mean"])
    out["z_jgdy_rate_30d"] = _zscore(out["jgdy_rate_30d"])
    out["z_jgdy_accel"] = _zscore(out["jgdy_accel_mean"])

    out["info_score"] = (
            0.45 * out["z_notice_rate_7d"] +
            0.20 * out["z_notice_accel"] +
            0.25 * out["z_jgdy_rate_30d"] +
            0.10 * out["z_jgdy_accel"]
    )

    out = out.sort_values("info_score", ascending=False).reset_index(drop=True)

    # 只对Top概念拉惩罚项，控制耗时
    top_m = max(CFG["top_concepts"] * 2, 30)
    concepts = out["概念"].head(top_m).tolist()
    rets = [concept_price_return(c) for c in concepts]
    funds = [concept_fund_20d(c) for c in concepts]

    tmp = pd.DataFrame({"概念": concepts, "concept_ret_20d": rets, "concept_fund_in_20d": funds})
    out = out.merge(tmp, on="概念", how="left")

    # ====== [替换] 过热惩罚：只惩罚“涨太多/流入太强”，不奖励下跌/流出 ======
    out["z_concept_ret_20d"] = _zscore(out["concept_ret_20d"])
    out["z_concept_fund_in_20d"] = _zscore(out["concept_fund_in_20d"])

    def relu(x):
        return np.maximum(x, 0)

    out["overheat"] = (
            0.55 * relu(out["z_concept_ret_20d"].fillna(0.0)) +
            0.45 * relu(out["z_concept_fund_in_20d"].fillna(0.0))
    )

    out["incubating_score"] = out["info_score"] - out["overheat"]

    # 可选：避免明显走弱的概念（你不想捡下跌垃圾）
    out = out[(out["concept_ret_20d"].isna()) | (out["concept_ret_20d"] > -0.08)].copy()

    return out


# ---------------- stock watchlist (avoid high) ----------------
def fetch_spot() -> pd.DataFrame:
    df = safe_ak_call(ak.stock_zh_a_spot_em)
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    df = df.rename(columns={
        pick_col(df, ["代码"]): "代码",
        pick_col(df, ["名称"]): "名称",
        pick_col(df, ["成交额", "成交额(元)"]): "成交额",
        pick_col(df, ["最新价", "最新", "现价"]): "最新价",
        pick_col(df, ["总市值", "市值", "总市值(元)"]): "总市值",
    })
    df["代码"] = df["代码"].apply(norm_code)
    df = df.dropna(subset=["代码"])
    df["成交额"] = pd.to_numeric(df.get("成交额", np.nan), errors="coerce")
    df["最新价"] = pd.to_numeric(df.get("最新价", np.nan), errors="coerce")
    df["总市值"] = pd.to_numeric(df.get("总市值", np.nan), errors="coerce")
    return df[["代码", "名称", "成交额", "总市值", "最新价"]]


def stock_hist_basic(code: str, days=260) -> pd.Series | None:
    end = datetime.now()
    start = end - timedelta(days=days)
    df = safe_ak_call(
        ak.stock_zh_a_hist,
        symbol=code, period="daily",
        start_date=_yyyymmdd(start),
        end_date=_yyyymmdd(end),
        adjust="qfq"
    )
    if df is None or len(df) < 80:
        return None

    close_col = pick_col(df, ["收盘", "close"])
    if not close_col:
        return None
    closes = pd.to_numeric(df[close_col], errors="coerce").dropna()
    if len(closes) < 80:
        return None

    close = float(closes.iloc[-1])
    ma60 = float(closes.tail(60).mean())
    over_ma60 = close / ma60 if ma60 > 0 else np.nan
    ret60 = float(closes.iloc[-1] / closes.iloc[-61] - 1) if len(closes) >= 61 else np.nan
    return pd.Series({"close": close, "ma60": ma60, "close_over_ma60": over_ma60, "ret60": ret60})


def build_stock_watchlist(top_concepts, members, spot, notice_feat, jgdy_feat) -> pd.DataFrame:
    mem = members[members["概念"].isin(top_concepts)].copy()
    mem["代码"] = mem["代码"].astype(str).str.zfill(6)

    base = mem.merge(spot, on="代码", how="left")
    if CFG["exclude_st"]:
        base = base[~base["名称"].astype(str).str.contains("ST", na=False)]
    base = base[pd.to_numeric(base["成交额"], errors="coerce").fillna(0) >= CFG["min_turnover"]].copy()

    base = base.merge(notice_feat, on="代码", how="left").merge(jgdy_feat, on="代码", how="left")
    for c in ["notice_7d", "notice_30d", "notice_lookback", "notice_accel",
              "jgdy_7d", "jgdy_30d", "jgdy_lookback", "jgdy_accel"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)

    out_rows = []
    for concept in top_concepts:
        sub = base[base["概念"] == concept].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(["成交额", "总市值"], ascending=[False, True]).head(CFG["max_stocks_per_concept"])

        for _, r in sub.iterrows():
            code = str(r["代码"]).zfill(6)
            try:
                st = stock_hist_basic(code, days=260)
                if st is None:
                    continue

                # 避免追高过滤
                if np.isfinite(st["ret60"]) and st["ret60"] > CFG["max_60d_return"]:
                    continue
                if np.isfinite(st["close_over_ma60"]) and st["close_over_ma60"] > CFG["max_close_over_ma60"]:
                    continue

                row = dict(r)
                row.update(st.to_dict())
                out_rows.append(row)
            except Exception:
                continue

    if not out_rows:
        return pd.DataFrame()

    w = pd.DataFrame(out_rows)
    w["z_notice"] = _zscore(w["notice_30d"])
    w["z_jgdy"] = _zscore(w["jgdy_30d"])
    w["z_ret60"] = _zscore(w["ret60"].fillna(0))
    w["z_over_ma60"] = _zscore((w["close_over_ma60"] - 1).fillna(0))
    w["stock_incubating_score"] = (0.55 * w["z_notice"] + 0.45 * w["z_jgdy"]) - (0.60 * w["z_ret60"] + 0.40 * w["z_over_ma60"])

    w = w.sort_values(["概念", "stock_incubating_score"], ascending=[True, False]).reset_index(drop=True)
    return w


def main():
    _ensure_dir(CFG["out_dir"])
    _ensure_dir(CFG["cache_dir"])

    print("AkShare:", getattr(ak, "__version__", "unknown"))
    print("开始：孕育期热点（长周期、领先信号）...")

    members = get_concept_members()
    print("概念成份:", len(members))

    notice_df = fetch_notices()
    notice_feat = stock_notice_features(notice_df)
    print("公告命中:", len(notice_feat))

    # 诊断：公告代码与概念成份代码交集
    member_codes = set(members["代码"].dropna().astype(str))
    notice_codes = set(notice_feat["代码"].dropna().astype(str)) if len(notice_feat) else set()
    overlap = len(member_codes & notice_codes)
    print(f"[diag] notice codes={len(notice_codes)}, member codes={len(member_codes)}, overlap={overlap}")

    jgdy_df = fetch_jgdy()
    jgdy_feat = stock_jgdy_features(jgdy_df)
    print("调研命中:", len(jgdy_feat))

    concept_df = build_concept_scores(members, notice_feat, jgdy_feat).sort_values("incubating_score", ascending=False)
    out1 = os.path.join(CFG["out_dir"], f"concept_incubator_{_yyyymmdd(datetime.now())}.csv")
    concept_df.head(300).to_csv(out1, index=False, encoding="utf-8-sig")
    print("输出概念孕育榜:", out1)

    top_concepts = concept_df.head(CFG["top_concepts"])["概念"].tolist()
    spot = fetch_spot()
    watch_df = build_stock_watchlist(top_concepts, members, spot, notice_feat, jgdy_feat)

    out2 = os.path.join(CFG["out_dir"], f"stock_watchlist_{_yyyymmdd(datetime.now())}.csv")
    if watch_df is None or len(watch_df) == 0:
        print("埋伏池为空：放宽 CFG['max_60d_return'] / CFG['max_close_over_ma60'] / CFG['min_turnover']")
        pd.DataFrame().to_csv(out2, index=False, encoding="utf-8-sig")
    else:
        watch_df = watch_df.groupby("概念", as_index=False, group_keys=False).head(20)
        watch_df.to_csv(out2, index=False, encoding="utf-8-sig")
        print("输出个股埋伏池:", out2)

    print("完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("程序异常：", repr(e))
        traceback.print_exc()
