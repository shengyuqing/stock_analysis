import math
from datetime import datetime

import akshare as ak
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

# ===================== 可调参数 =====================

PE_MAX = 40                 # 成长股估值上限（TTM）
LIQUID_RATIO_MIN = 1.2      # 流动比率下限
DEBT_NCAV_MAX = 1.5         # 债务 / 净流动资产上限
MV_TANGIBLE_MAX = 2.5       # 市值 / 有形资产上限
PROFIT_CAGR_MIN = 0.15      # 5 年净利 CAGR 下限 15%
PEG_MAX = 1.5               # PEG 上限
REQUIRE_DIVIDEND = False    # 是否强制要求近 5 年有现金分红

# ===================== 工具函数 =====================

def add_market_prefix(stock_code: str) -> str:
    """
    6 位代码 -> 带交易所前缀
    6xxxx / 9xxxx -> SH
    0xxxx / 3xxxx -> SZ
    8xxxxxx        -> BJ
    """
    stock_code = stock_code.strip()
    if stock_code.startswith(('6', '9')):
        return 'SH' + stock_code
    elif stock_code.startswith(('0', '3')):
        return 'SZ' + stock_code
    elif stock_code.startswith('8'):
        return 'BJ' + stock_code
    else:
        raise ValueError(f"无法识别股票代码的市场前缀：{stock_code}")


def get_pe_mv_baidu(stock_code: str):
    """
    从百度估值接口获取：
    - 市盈率(TTM)
    - 总市值
    若失败或数据缺失，返回 (None, None)
    """
    try:
        pe_df = ak.stock_zh_valuation_baidu(
            symbol=stock_code,
            indicator="市盈率(TTM)",
            period="近一年",
        )
    except Exception as e:
        print(f"[PE] 获取失败 {stock_code}: {e}")
        return None, None

    if pe_df.empty or "value" not in pe_df.columns:
        return None, None

    pe_series = pd.to_numeric(pe_df["value"], errors="coerce").dropna()
    if pe_series.empty:
        return None, None

    pe_ttm = float(pe_series.iloc[-1])

    # 获取总市值
    try:
        mv_df = ak.stock_zh_valuation_baidu(
            symbol=stock_code,
            indicator="总市值",
            period="近一年",
        )
    except Exception as e:
        print(f"[市值] 获取失败 {stock_code}: {e}")
        return pe_ttm, None

    if mv_df.empty or "value" not in mv_df.columns:
        return pe_ttm, None

    mv_series = pd.to_numeric(mv_df["value"], errors="coerce").dropna()
    if mv_series.empty:
        return pe_ttm, None

    total_mv = float(mv_series.iloc[-1])
    return pe_ttm, total_mv


def get_current_ratio_ths(stock_code: str, min_ratio: float = 1.2):
    """
    使用同花顺主要指标接口，取最后一个非空的“流动比率”
    返回: (是否达标, 报告期, 流动比率)
    """
    try:
        df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按单季度")
    except Exception as e:
        print(f"[流动比率] 获取失败 {stock_code}: {e}")
        return False, None, None

    if df.empty or "流动比率" not in df.columns:
        return False, None, None

    df = df.copy()
    df["流动比率"] = (
        df["流动比率"]
        .replace(["", "-"], pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df_valid = df[df["流动比率"].notna()]
    if df_valid.empty:
        return False, None, None

    last = df_valid.iloc[-1]
    date_col = "报告期" if "报告期" in df_valid.columns else "报告日期"
    ratio = float(last["流动比率"])
    date_value = last[date_col]

    return ratio >= min_ratio, date_value, ratio


def check_debt_ncav(balance_df: pd.DataFrame, max_ratio: float = 1.5):
    """
    使用东方财富-资产负债表按年度数据，检查：
    债务 / 净流动资产 <= max_ratio
    返回: (是否达标, 实际比值)
    """
    if balance_df is None or balance_df.empty:
        return False, None

    first = balance_df.iloc[0]

    def _get_float(val):
        try:
            f = float(val)
            if math.isnan(f):
                return 0.0
            return f
        except Exception:
            return 0.0

    total_current_assets = _get_float(first.get("TOTAL_CURRENT_ASSETS", 0))
    total_liabilities = _get_float(first.get("TOTAL_LIABILITIES", 0))

    net_current_assets = total_current_assets - total_liabilities
    if total_liabilities <= 0 or net_current_assets <= 0:
        return False, None

    debt_to_ncav = total_liabilities / net_current_assets
    ok = 0 < debt_to_ncav <= max_ratio
    return ok, debt_to_ncav


def calc_tangible_assets_ratio(balance_df: pd.DataFrame, total_mv: float):
    """
    计算有形资产总额和市值/有形资产比值
    返回: (tangible_assets, mv_to_tangible) 或 (None, None)
    """
    if balance_df is None or balance_df.empty or total_mv is None:
        return None, None

    first = balance_df.iloc[0]

    def _get_float(val):
        try:
            f = float(val)
            if math.isnan(f):
                return 0.0
            return f
        except Exception:
            return 0.0

    cols = [
        "FIXED_ASSET",
        "CIP",
        "OIL_GAS_ASSET",
        "PRODUCTIVE_BIOLOGY_ASSET",
        "INVENTORY",
        "MONETARYFUNDS",
        "USERIGHT_ASSET",
        "PROJECT_MATERIAL",
    ]

    tangible_assets = 0.0
    for col in cols:
        if col in first.index:
            tangible_assets += _get_float(first[col])

    if tangible_assets <= 0:
        return None, None

    mv_to_tangible = total_mv / tangible_assets
    return tangible_assets, mv_to_tangible


def calc_profit_growth(full_code: str, required_years: int = 5):
    """
    使用东方财富-利润表按年度数据，计算 5 年净利成长性：
    - 取最近 required_years 年
    - 检查是否全部 > 0
    - 计算 CAGR、最近一年 vs 5 年均值、最近一年 vs 上一年
    返回 dict 或 None
    """
    try:
        df = ak.stock_profit_sheet_by_yearly_em(symbol=full_code)
    except Exception as e:
        print(f"[利润表] 获取失败 {full_code}: {e}")
        return None

    if df is None or df.empty or "REPORT_DATE" not in df.columns:
        return None

    df = df.copy()
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce").dt.date
    df = df.dropna(subset=["REPORT_DATE"]).sort_values("REPORT_DATE")

    # 自动识别“净利润”列
    candidates = []
    for col in df.columns:
        upper = col.upper()
        if "PARENT_NETPROFIT" in upper:
            candidates.insert(0, col)
        elif "NETPROFIT" in upper or "净利润" in col:
            candidates.append(col)

    if not candidates:
        return None

    net_col = candidates[0]
    df["NET_PROFIT"] = pd.to_numeric(df[net_col], errors="coerce")
    df = df.dropna(subset=["NET_PROFIT"])

    if df.empty:
        return None

    # 取最近 required_years 个年度
    df_tail = df.tail(required_years)
    if len(df_tail) < required_years:
        return None

    years = [d.year for d in df_tail["REPORT_DATE"]]
    profits = df_tail["NET_PROFIT"].tolist()

    if min(profits) <= 0:
        return {
            "ok": False,
            "reason": "negative_profit",
            "years": years,
            "profits": profits,
        }

    first = profits[0]
    last = profits[-1]
    n = len(profits)

    if first <= 0 or n < 2:
        return None

    cagr = (last / first) ** (1 / (n - 1)) - 1  # 年复合增长率
    avg5 = sum(profits) / n
    last_vs_avg = last / avg5 if avg5 != 0 else float("inf")
    last_vs_prev = last / profits[-2] if profits[-2] != 0 else float("inf")

    return {
        "ok": True,
        "years": years,
        "profits": profits,
        "cagr": cagr,
        "last_vs_avg": last_vs_avg,
        "last_vs_prev": last_vs_prev,
    }


def has_dividend_last_5_years(stock_code: str, min_positive_years: int = 3):
    """
    使用历史分红接口，判断过去 5 年是否有至少 min_positive_years 年发生现金分红。
    若数据缺失，返回 None（不作为淘汰条件）。
    """
    try:
        df = ak.stock_history_dividend_detail(stock=stock_code)
    except Exception as e:
        print(f"[分红] 获取失败 {stock_code}: {e}")
        return None

    if df is None or df.empty:
        return None

    df = df.copy()

    # 年份列
    if "分红年度" in df.columns:
        years = df["分红年度"].astype(str).str[:4]
    elif "报告期" in df.columns:
        years = df["报告期"].astype(str).str[:4]
    else:
        return None

    df["year"] = years

    # 现金分红列
    cash_cols = [c for c in df.columns if ("现金" in c) or ("派息" in c)]
    if not cash_cols:
        return None
    cash_col = cash_cols[0]
    df[cash_col] = pd.to_numeric(df[cash_col], errors="coerce").fillna(0)

    current_year = datetime.now().year
    target_years = [str(y) for y in range(current_year - 5, current_year)]
    df_5y = df[df["year"].isin(target_years)]
    if df_5y.empty:
        return False

    grouped = df_5y.groupby("year")[cash_col].sum()
    positive_years = (grouped > 0).sum()
    return positive_years >= min_positive_years


# ===================== 主筛选函数 =====================

def find_growth_stocks():
    # 获取全部 A 股代码和名称
    code_df = ak.stock_info_a_code_name()
    # 兼容不同版本列名
    if "code" in code_df.columns:
        code_df = code_df.rename(columns={"code": "代码", "name": "名称"})
    elif "代码" not in code_df.columns or "名称" not in code_df.columns:
        raise RuntimeError("无法识别 stock_info_a_code_name 返回的列名，请打印查看后手工修改。")

    results = []

    for idx, row in code_df.iterrows():
        stock_code = str(row["代码"]).strip().zfill(6)
        stock_name = str(row["名称"]).strip()
        full_code = add_market_prefix(stock_code)

        try:
            # 1) 估值：PE & 总市值
            pe_ttm, total_mv = get_pe_mv_baidu(stock_code)
            if pe_ttm is None or pe_ttm <= 0 or pe_ttm > PE_MAX:
                continue

            # 2) 流动比率
            liq_ok, liq_date, liq_value = get_current_ratio_ths(stock_code, LIQUID_RATIO_MIN)
            if not liq_ok:
                continue

            # 3) 资产负债表：负债 / 净流动资产、有形资产
            try:
                balance_df = ak.stock_balance_sheet_by_yearly_em(symbol=full_code)
            except Exception as e:
                print(f"[资产负债表] 获取失败 {full_code}: {e}")
                continue

            debt_ok, debt_ratio = check_debt_ncav(balance_df, DEBT_NCAV_MAX)
            if not debt_ok:
                continue

            tangible_assets, mv_tangible = calc_tangible_assets_ratio(balance_df, total_mv)
            if tangible_assets is None or mv_tangible is None:
                continue
            if mv_tangible > MV_TANGIBLE_MAX:
                continue

            # 4) 盈利成长性（5 年）
            profit_info = calc_profit_growth(full_code, required_years=5)
            if not profit_info or not profit_info.get("ok", False):
                continue
            if profit_info["cagr"] < PROFIT_CAGR_MIN:
                continue
            if profit_info["last_vs_avg"] < 1.0:   # 最近一年 > 5 年平均
                continue
            if profit_info["last_vs_prev"] < 1.0:  # 最近一年 >= 上一年
                continue

            # 5) 分红（可选）
            if REQUIRE_DIVIDEND:
                div_ok = has_dividend_last_5_years(stock_code, min_positive_years=3)
                if div_ok is False:
                    continue

            # 6) PEG
            growth_pct = profit_info["cagr"] * 100
            peg = pe_ttm / growth_pct if growth_pct > 0 else None
            if peg is not None and peg > PEG_MAX:
                continue

            results.append({
                "代码": stock_code,
                "名称": stock_name,
                "PE_TTM": round(pe_ttm, 2),
                "PEG": round(peg, 2) if peg is not None else None,
                "净利CAGR_5Y(%)": round(profit_info["cagr"] * 100, 1),
                "近年净利/5年均值": round(profit_info["last_vs_avg"], 2),
                "近年净利/上一年": round(profit_info["last_vs_prev"], 2),
                "流动比率": round(liq_value, 2) if liq_value is not None else None,
                "债务/净流动资产": round(debt_ratio, 2) if debt_ratio is not None else None,
                "市值/有形资产": round(mv_tangible, 2),
                "总市值": round(total_mv, 0) if total_mv is not None else None,
                "有形资产": round(tangible_assets, 0),
            })

            print(f"筛选通过: {stock_code} {stock_name}")

        except Exception as e:
            # 单只股票出错，打印后继续
            print(f"[错误] 处理 {stock_code} {stock_name} 失败: {e}")
            continue

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        # 按 5 年净利增速高 -> PE 低 排序
        result_df = result_df.sort_values(
            by=["净利CAGR_5Y(%)", "PE_TTM"],
            ascending=[False, True],
        )
    return result_df


if __name__ == "__main__":
    df = find_growth_stocks()
    if df.empty:
        print("没有找到符合高成长性条件的股票")
    else:
        out_file = "高成长性_优质股票筛选结果.xlsx"
        df.to_excel(out_file, index=False)
        print(f"共找到 {len(df)} 只股票，已保存到 {out_file}")
        print(df.head(20))
