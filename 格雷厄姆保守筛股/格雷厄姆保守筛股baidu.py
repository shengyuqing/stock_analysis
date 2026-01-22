import akshare as ak
import pandas as pd
from datetime import datetime
import time

pd.set_option('future.no_silent_downcasting', True)


# ------------ 工具函数 ------------

def add_market_prefix(stock_code: str) -> str:
    """
    返回 SH000001 / SZ000001 / BJ8xxxxx 这种格式（目前只在日志里用，不给接口用）
    """
    stock_code = stock_code.strip()
    if stock_code.startswith(('6', '9')):     # 上交所
        return 'SH' + stock_code
    elif stock_code.startswith(('0', '3')):   # 深交所
        return 'SZ' + stock_code
    elif stock_code.startswith('8'):          # 北交所
        return 'BJ' + stock_code
    else:
        raise ValueError(f"无法识别股票代码的市场前缀：{stock_code}")


def code_to_sina(stock_code: str) -> str:
    """
    新浪接口需要 sh600000 / sz000001 这种格式
    """
    stock_code = stock_code.strip()
    if stock_code.startswith(('6', '9')):
        return 'sh' + stock_code
    else:
        return 'sz' + stock_code


def get_all_a_share_basic() -> pd.DataFrame:
    """
    用交易所接口获取 A 股代码和名称（不走东财）
    返回列：代码（6位）、名称
    """
    df = ak.stock_info_a_code_name()      # A 股股票代码和简称（交易所）
    df['代码'] = df['code'].str.extract(r'(\d{6})')
    df['名称'] = df['name']
    df = df.dropna(subset=['代码'])
    return df[['代码', '名称']]


def get_baidu_pe_and_mv(stock_code: str):
    """
    从 百度股市通 获取单个股票的：
    - 市盈率(TTM) 作为“市盈率-动态”
    - 总市值
    """
    # 市盈率(TTM)
    pe_df = ak.stock_zh_valuation_baidu(symbol=stock_code, indicator="市盈率(TTM)")
    if pe_df.empty or 'value' not in pe_df.columns:
        raise RuntimeError("百度估值-市盈率(TTM) 数据为空")
    pe_df = pe_df.dropna(subset=['value'])
    pe_ttm = float(pe_df.iloc[-1]['value'])

    # 总市值
    mv_df = ak.stock_zh_valuation_baidu(symbol=stock_code, indicator="总市值")
    if mv_df.empty or 'value' not in mv_df.columns:
        raise RuntimeError("百度估值-总市值 数据为空")
    mv_df = mv_df.dropna(subset=['value'])
    total_mv = float(mv_df.iloc[-1]['value'])

    # 注意：这里 total_mv 的单位要跟有形资产保持一致，下面只做相对比较，不考虑绝对单位
    return pe_ttm, total_mv


# ------------ 条件函数：流动比率 / 资产负债表 ------------

def liudongbi_func1(stock_code):
    """
    使用同花顺关键指标接口：
    取“按单季度”的流动比率，选最后一个非空值。
    返回：是否 >=1.5, 对应报告期, 流动比率数值
    """
    df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按单季度")

    if '流动比率' not in df.columns:
        return False, None, None

    df['流动比率'] = df['流动比率'].replace(['', '-'], pd.NA)
    df['流动比率'] = pd.to_numeric(df['流动比率'], errors='coerce').astype('float64')

    df_valid = df[df['流动比率'].notna()]
    if df_valid.empty:
        return False, None, None

    last = df_valid.iloc[-1]
    date_col = '报告期' if '报告期' in df.columns else ('报告日期' if '报告日期' in df.columns else None)
    liudongbi = float(last['流动比率'])
    liudongbi_date = last[date_col] if date_col is not None else None

    result = liudongbi >= 1.5
    return result, liudongbi_date, liudongbi


def load_balance_sheet_sina(stock_code: str) -> pd.DataFrame:
    """
    用 新浪 财务报表-资产负债表 替代原来的东财资产负债表
    """
    sina_code = code_to_sina(stock_code)
    df = ak.stock_financial_report_sina(stock=sina_code, symbol="资产负债表")
    if df.empty:
        raise RuntimeError("新浪资产负债表为空")

    if '报告日' in df.columns:
        df = df.sort_values('报告日', ascending=False).reset_index(drop=True)
    return df


def zhaiwu_lidong_func_sina(stock_code, balance_df, thresh=1.1):
    """
    使用新浪资产负债表计算：
    debt_to_net_current_asset = 总负债 / (流动资产 - 总负债)
    要求 0 < 指标 <= thresh
    """
    row = balance_df.iloc[0]

    def safe_get(col_names, default=None):
        for c in col_names:
            if c in row and pd.notna(row[c]):
                return float(row[c])
        return default

    total_current_assets = safe_get(['流动资产', '流动资产合计'])
    total_liabilities = safe_get(['负债合计', '负债总计'])

    if total_current_assets is None or total_liabilities is None:
        return False, None

    denom = total_current_assets - total_liabilities
    if denom <= 0:
        return False, None

    debt_to_ncav = total_liabilities / denom

    if 0 < debt_to_ncav <= thresh:
        return True, debt_to_ncav
    else:
        return False, debt_to_ncav


# ------------ 条件函数：有形资产 ------------

def youxingzichan_sina(total_market_value, balance_df, thresh=1.3):
    """
    用新浪资产负债表估算“有形资产”：
    固定资产 + 在建工程 + 油气资产 + 生产性生物资产 + 存货 + 货币资金 + 使用权资产 + 工程物资
    要求： 市值 / 有形资产 < thresh
    """
    row = balance_df.iloc[0]

    def safe_val(col):
        return float(row[col]) if col in row and pd.notna(row[col]) else 0.0

    tangible_assets = (
        safe_val('固定资产')
        + safe_val('在建工程')
        + safe_val('油气资产')
        + safe_val('生产性生物资产')
        + safe_val('存货')
        + safe_val('货币资金')
        + safe_val('使用权资产')
        + safe_val('工程物资')
    )

    if tangible_assets <= 0:
        return False, None

    ratio = total_market_value / tangible_assets
    return ratio < thresh, ratio


# ------------ 条件函数：分红 ------------

def check_recent_dividend_sina(stock_code: str, years: int = 5) -> bool:
    """
    使用新浪分红配股接口，检查最近 years 年是否每年都有现金分红 > 0
    """
    try:
        df = ak.stock_history_dividend_detail(symbol=stock_code, indicator="分红")
    except Exception:
        return False

    if df.empty or '公告日期' not in df.columns:
        return False

    df = df.copy()
    df['year'] = df['公告日期'].astype(str).str[:4].astype(int)

    dividend_cols = [c for c in df.columns if '每股派现' in c or '派息' in c or '派现' in c]
    if not dividend_cols:
        return False
    div_col = dividend_cols[0]

    current_year = datetime.now().year
    for year in range(current_year - 1, current_year - years - 1, -1):
        sub = df[df['year'] == year]
        if sub.empty:
            return False
        total_div = pd.to_numeric(sub[div_col], errors='coerce').fillna(0).sum()
        if total_div <= 0:
            return False

    return True


# ------------ 新增：净利润 5 年为正 + 最近一年大于过去 5 年平均 ------------

def load_profit_sheet_sina(stock_code: str) -> pd.DataFrame:
    """
    新浪 财务报表-利润表
    """
    sina_code = code_to_sina(stock_code)
    df = ak.stock_financial_report_sina(stock=sina_code, symbol="利润表")
    if df.empty:
        raise RuntimeError("新浪利润表为空")
    if '报告日' in df.columns:
        df['报告日'] = pd.to_datetime(df['报告日'], errors='coerce')
        df = df.dropna(subset=['报告日'])
        # 只要年报（12-31）
        df = df[(df['报告日'].dt.month == 12) & (df['报告日'].dt.day == 31)]
        df = df.sort_values('报告日', ascending=False).reset_index(drop=True)
    return df


def get_recent_profit_series(stock_code: str, max_years: int = 6) -> pd.Series:
    """
    取最近 max_years 年的年报净利润序列（按时间从近到远）
    返回一个 Series，index 为年份，value 为净利润
    """
    df = load_profit_sheet_sina(stock_code)

    if df.empty:
        raise RuntimeError("利润表（年报）为空")

    # 找一个“净利润”列
    profit_cols = [c for c in df.columns if '净利润' in c]
    if not profit_cols:
        raise RuntimeError("利润表中找不到含“净利润”的列")
    col = profit_cols[0]

    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[col])

    df = df.head(max_years)
    s = df[col]
    s.index = df['报告日'].dt.year
    return s


def check_profit_conditions(stock_code: str):
    """
    返回：
    stable_ok: 过去5年净利润都 > 0 （不含最近一年）
    growth_ok: 最近一年净利润 > 过去5年平均净利润
    last_profit: 最近一年净利润
    avg_5y_profit: 过去5年平均净利润
    """
    s = get_recent_profit_series(stock_code, max_years=6)
    if len(s) < 6:
        # 数据不足 6 年，按不满足处理
        return False, False, None, None

    latest = float(s.iloc[0])
    prev5 = s.iloc[1:6].astype(float)

    stable_ok = (prev5 > 0).all()
    avg_5y = prev5.mean()
    growth_ok = latest > avg_5y

    return stable_ok, growth_ok, latest, avg_5y


# ------------ 主函数：按格雷厄姆思路选股 ------------

def get_qualified_stocks():
    # 1. 获取 A 股代码与名称（不经东财）
    basic_df = get_all_a_share_basic()

    qualified_stocks = []

    for _, row in basic_df.iterrows():
        stock_code = row['代码']
        stock_name = row['名称']

        try:
            # (1) 市盈率：用百度 PE(TTM)，要求 0 < PE <= 9
            pe_ttm, total_market_value = get_baidu_pe_and_mv(stock_code)
            if not (0 < pe_ttm <= 9):
                continue

            # (2) 流动比率 >= 1.5
            ok_liudongbi, liudongbi_date, liudongbi_val = liudongbi_func1(stock_code)
            if not ok_liudongbi:
                continue

            # (3) 债务 / 净流动资产 <= 110%（0 < 指标 <= 1.1）
            balance_df = load_balance_sheet_sina(stock_code)
            ok_debt, debt_to_ncav = zhaiwu_lidong_func_sina(stock_code, balance_df, thresh=1.1)
            if not ok_debt:
                continue

            # (4) 盈利稳定性：过去五年净利润都 > 0（不含最近一年）
            # (6) 盈利增长：最近一年净利润 > 过去五年平均净利润
            stable_ok, growth_ok, last_profit, avg_5y_profit = check_profit_conditions(stock_code)
            if not stable_ok:
                # 过去 5 年里有亏损
                continue
            if not growth_ok:
                # 最近一年没超过过去 5 年平均
                continue

            # (5) 分红：过去 5 年每年都有现金分红 > 0
            ok_dividend = check_recent_dividend_sina(stock_code, years=5)
            if not ok_dividend:
                continue

            # (7) 价格 / 有形资产（这里用 市值 / 有形资产） <= 1.3（你可以改成 1.2）
            ok_tangible, mv_over_tangible = youxingzichan_sina(total_market_value, balance_df, thresh=1.3)
            if not ok_tangible:
                continue

            print(f"{stock_code} {stock_name} 满足全部条件，市值/有形资产={mv_over_tangible:.2f}")

            qualified_stocks.append({
                '代码': stock_code,
                '名称': stock_name,
                '市盈率-动态(PE_TTM)': pe_ttm,
                '最新流动比率': liudongbi_val,
                '流动比率报告期': liudongbi_date,
                '债务/净流动资产': debt_to_ncav,
                '最近一年净利润': last_profit,
                '过去5年平均净利润': avg_5y_profit,
                '市值/有形资产': mv_over_tangible,
                '总市值(百度)': total_market_value,
            })

            # 稍微限速，防止接口被封，可以视情况调小 / 去掉
            time.sleep(0.4)

        except Exception as e:
            print(f"{stock_code} {stock_name} 出错，已跳过：{e}")
            continue

    return pd.DataFrame(qualified_stocks)


if __name__ == "__main__":
    result = get_qualified_stocks()

    if not result.empty:
        out_name = "符合条件的A股股票_格雷厄姆完整版.xlsx"
        result.to_excel(out_name, index=False)
        print(f"共找到 {len(result)} 只符合条件的股票，已保存到：{out_name}")
        print(result.head())
    else:
        print("没有找到符合条件的股票")
