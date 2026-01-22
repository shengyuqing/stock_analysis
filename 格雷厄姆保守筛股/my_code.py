import akshare as ak
import pandas as pd
from datetime import datetime
import os
import time

pd.set_option('future.no_silent_downcasting', True)


def format_stock_code(df: pd.DataFrame) -> pd.DataFrame:
    # 类型安全转换
    df['代码'] = df['代码'].astype(str).str.strip()

    # 智能补零逻辑
    def _zero_pad(code):
        if code.startswith('8') and len(code) == 7:  # 北交所特殊处理
            return code.zfill(7)
        elif len(code) < 6:  # 沪深市场补零
            return code.zfill(6)
        return code  # 已有合规代码

    df['代码'] = df['代码'].apply(_zero_pad)
    return df


def add_market_prefix(stock_code: str) -> str:
    stock_code = stock_code.strip()
    if stock_code.startswith(('6', '9')):     # 上交所
        return 'SH' + stock_code
    elif stock_code.startswith(('0', '3')):   # 深交所
        return 'SZ' + stock_code
    elif stock_code.startswith('8'):          # 北交所
        return 'BJ' + stock_code
    else:
        raise ValueError(f"无法识别股票代码的市场前缀：{stock_code}")


def liudongbi_func1(stock_code):
    """
    从同花顺摘要里取按单季度的流动比率，拿最后一个非空值。
    返回: (是否 >= 1.5, 对应报告期日期)
    """
    try:
        df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按单季度")
    except Exception as e:
        print(f"{stock_code} 获取流动比率失败: {e}")
        return False, None

    if df is None or df.empty or '流动比率' not in df.columns:
        print(f"{stock_code} 没有流动比率数据")
        return False, None

    # 先把空字符串／“-” 替换为 NA
    df['流动比率'] = df['流动比率'].replace(['', '-'], pd.NA)
    # 转为 float
    df['流动比率'] = pd.to_numeric(df['流动比率'], errors='coerce').astype('float64')

    # 过滤出最后一个非空的“流动比率”
    df_valid = df[df['流动比率'].notna()]

    if df_valid.empty:
        print(f"{stock_code} 没有找到任何非空的流动比率")
        return False, None

    last = df_valid.iloc[-1]
    date_col = '报告期' if '报告期' in df.columns else ('报告日期' if '报告日期' in df.columns else None)
    liudongbi = last['流动比率']
    liudongbi_date = last[date_col] if date_col else None

    if liudongbi is None or pd.isna(liudongbi):
        return False, liudongbi_date

    if liudongbi >= 1.5:
        return True, liudongbi_date
    else:
        return False, liudongbi_date


def zhaiwu_lidong_func(stock_code, stock_balance_sheet, thresh=1.1):
    """
    计算: 总负债 / (流动资产 - 总负债)，要求在 (0, thresh] 之间
    """
    if stock_balance_sheet is None or stock_balance_sheet.empty:
        print(f"{stock_code} 资产负债表为空，无法计算债务/净流动资产")
        return False

    first_row = stock_balance_sheet.iloc[0]

    total_current_assets = first_row.get('TOTAL_CURRENT_ASSETS', None)
    total_liabilities = first_row.get('TOTAL_LIABILITIES', None)

    if pd.isna(total_current_assets) or pd.isna(total_liabilities):
        print(f"{stock_code} 资产负债表缺少必要字段")
        return False

    denom = total_current_assets - total_liabilities
    if denom <= 0:
        # 净流动资产 <= 0，直接淘汰
        return False

    debt_to_net_current_asset = total_liabilities / denom

    if 0 < debt_to_net_current_asset <= thresh:
        return True
    else:
        return False


def youxingzichan(stock_code, full_code, spot_data, stock_balance_sheet, thresh):
    """
    用市值 / 有形资产 的比值做筛选，要求 < thresh
    有形资产 = 多个科目之和
    """
    # 现在市值
    target_row = spot_data[spot_data["代码"] == stock_code]
    if target_row.empty:
        print(f"{stock_code} 在实时行情里找不到，跳过有形资产判断")
        return False

    total_market_value = target_row["总市值"].iloc[0]
    if pd.isna(total_market_value) or total_market_value <= 0:
        print(f"{stock_code} 总市值异常，跳过")
        return False

    if stock_balance_sheet is None or stock_balance_sheet.empty:
        print(f"{stock_code} 资产负债表为空，无法计算有形资产")
        return False

    first_row = stock_balance_sheet.iloc[0]

    cols = [
        "FIXED_ASSET", "CIP", "OIL_GAS_ASSET",
        "PRODUCTIVE_BIOLOGY_ASSET", "INVENTORY",
        "MONETARYFUNDS", "USERIGHT_ASSET", "PROJECT_MATERIAL"
    ]
    exist_cols = [c for c in cols if c in first_row.index]

    if not exist_cols:
        print(f"{stock_code} 没有有形资产相关科目，跳过")
        return False

    tangible_assets = first_row[exist_cols].fillna(0).sum()

    if tangible_assets <= 0:
        print(f"{stock_code} 有形资产总额<=0，跳过")
        return False

    ratio = total_market_value / tangible_assets
    if ratio < thresh:
        return True
    else:
        return False


def recent_dividend(stock_code, full_code, stock_balance_sheet):
    """
    打印最近若干年度的现金分红股息率，不作为硬性筛选条件。
    """
    try:
        stock_fhps_detail_em_df = ak.stock_fhps_detail_em(symbol=stock_code)
    except Exception as e:
        print(f"{stock_code} 获取分红数据失败: {e}")
        return

    if stock_fhps_detail_em_df is None or stock_fhps_detail_em_df.empty:
        print(f"{stock_code} 没有分红数据")
        return

    if "报告期" not in stock_fhps_detail_em_df.columns or "现金分红-股息率" not in stock_fhps_detail_em_df.columns:
        print(f"{stock_code} 分红数据缺少必要字段")
        return

    target_dates = ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31', '2020-12-31']
    date_objects = [d.date() for d in pd.to_datetime(target_dates)]
    df_selected = stock_fhps_detail_em_df[stock_fhps_detail_em_df["报告期"].isin(date_objects)]

    if df_selected.empty:
        print(f"{stock_code} 最近五个年度没有匹配到固定 12-31 报告期的分红记录")
        return

    print(df_selected[["报告期", "现金分红-股息率"]].sort_values("报告期", ascending=False).to_string(index=False))


def load_spot_data(use_cache_first=True,
                   cache_path="akshare_output/stocks.csv",
                   max_retries=3,
                   sleep_seconds=3):
    """
    带重试 + 本地缓存的行情获取函数。
    """
    if use_cache_first and os.path.exists(cache_path):
        print("优先从本地缓存读取 A 股行情...")
        df = pd.read_csv(cache_path, dtype={"代码": str})
        return df

    # 没缓存就在线获取 + 重试
    for i in range(max_retries):
        try:
            print(f"第 {i + 1} 次尝试从 akshare 获取 A 股行情数据...")
            df = ak.stock_zh_a_spot_em()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"已保存行情缓存到 {cache_path}")
            return df
        except Exception as e:
            print(f"第 {i + 1} 次获取行情失败: {e}")
            if i < max_retries - 1:
                time.sleep(sleep_seconds)
            else:
                print("重试次数已用完，无法获取实时行情数据。")
                return pd.DataFrame()


def get_spot_data_from_em():
    # 上证
    try:
        sh = ak.stock_sh_a_spot_em()
    except Exception as e:
        print("获取上证 A 股行情失败:", e)
        sh = pd.DataFrame()

    # 深证
    try:
        sz = ak.stock_sz_a_spot_em()
    except Exception as e:
        print("获取深证 A 股行情失败:", e)
        sz = pd.DataFrame()

    if sh.empty and sz.empty:
        print("上证和深证行情都获取失败")
        return pd.DataFrame()

    df = pd.concat([sh, sz], ignore_index=True)
    # 统一一下列名，如果和原来的 stock_zh_a_spot_em 有差异，需要对照文档改
    df = format_stock_code(df)
    return df


def get_qualified_stocks():
    # 获取所有A股实时行情数据（包含动态市盈率）
    spot_data = load_spot_data(use_cache_first=False)
    #spot_data = get_spot_data_from_em()
    if spot_data.empty:
        print("无法获取A股实时行情数据（网络+缓存都失败）")
        return pd.DataFrame()

    spot_data = format_stock_code(spot_data)

    # 筛选动态市盈率 <= 9 的股票
    if '市盈率-动态' not in spot_data.columns:
        print("行情数据中没有 '市盈率-动态' 列")
        return pd.DataFrame()

    filtered = spot_data[(spot_data['市盈率-动态'] <= 9) & (spot_data['市盈率-动态'] > 0)].copy()

    qualified_stocks = []

    for _, row in filtered.iterrows():
        stock_code = str(row['代码']).strip()
        stock_name = row['名称']

        try:
            full_code = add_market_prefix(stock_code)
        except ValueError as e:
            print(e)
            continue

        # 2、流动比率 >= 1.5
        ok_liudongbi, liudongbi_date = liudongbi_func1(stock_code)
        if not ok_liudongbi:
            continue

        # 3、(债务/净流动资产价值) <= 110%
        try:
            stock_balance_sheet = ak.stock_balance_sheet_by_yearly_em(symbol=full_code)
        except Exception as e:
            print(f"{stock_code} 获取资产负债表失败: {e}")
            continue

        if stock_balance_sheet is None or stock_balance_sheet.empty:
            print(f"{stock_code} 资产负债表为空，跳过")
            continue

        if not zhaiwu_lidong_func(stock_code, stock_balance_sheet, thresh=1.1):
            continue

        # 4、有形资产
        if not youxingzichan(stock_code, full_code, spot_data, stock_balance_sheet, thresh=1.3):
            print(stock_name, "有形资产不满足")
            continue
        else:
            print(stock_name, "恭喜，有形资产满足！")

        # 5、打印过去5年分红信息（不作为硬筛选）
        try:
            recent_dividend(stock_code, full_code, stock_balance_sheet)
        except Exception as e:
            print(f"{stock_code} 打印分红信息失败: {e}")

        # 添加到结果列表（这里先只保存基础信息，有需要再加别的指标）
        qualified_stocks.append({
            '代码': stock_code,
            '名称': stock_name,
            '市盈率-动态': row['市盈率-动态'],
            '最新价': row.get('最新价', None),
            '总市值': row.get('总市值', None),
            '最近流动比率报告期': liudongbi_date,
        })
        print(f"找到符合条件的股票: {stock_code} {stock_name}")

    return pd.DataFrame(qualified_stocks)


if __name__ == "__main__":
    result = get_qualified_stocks()

    # 保存结果
    if not result.empty:
        outfile = "符合条件的A股股票.xlsx"
        result.to_excel(outfile, index=False)
        print(f"共找到 {len(result)} 只符合条件的股票，已保存到 {outfile}")
        print(result)
    else:
        print("没有找到符合条件的股票")
