import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
pd.set_option('future.no_silent_downcasting',  True)
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
    if stock_code.startswith(('6', '9')):     # 上交所：600xxx, 601xxx, 603xxx, 688xxx, 900xxx 等
        return 'SH' + stock_code
    elif stock_code.startswith(('0', '3')):   # 深交所：000xxx, 002xxx, 300xxx 等
        return 'SZ' + stock_code
    elif stock_code.startswith('8'):          # 北交所：830xxx, 831xxx 等
        return 'BJ' + stock_code
    else:
        raise ValueError(f"无法识别股票代码的市场前缀：{stock_code}")

def liudongbi_func1(stock_code):
    df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按单季度")
    # 先把空字符串／“-” 替换为 NA
    df['流动比率'] = df['流动比率'].replace(['', '-'], pd.NA)
    # 先用 to_numeric 把可转的变成数值（此时可能是 Int64 或 Float64），再强制转成 float
    df['流动比率'] = pd.to_numeric(df['流动比率'], errors='coerce').astype('float64')
    # 2. 过滤出最后一个非空的“流动比率”
    df_valid = df[df['流动比率'].notna()]
    # 3. 取最后一行
    if not df_valid.empty:
        last = df_valid.iloc[-1]
        # 假设日期列叫 “报告期” 或 “报告日期”，请根据你的实际列名替换
        date_col = '报告期' if '报告期' in df.columns else '报告日期'
        # print("最后一次非空流动比率：", last['流动比率'])
        # print(f"对应的{date_col}：", last[date_col])
        liudongbi = last['流动比率']
        liudongbi_date = last[date_col]
        print('流动比率：',liudongbi, liudongbi_date)
        if liudongbi >= 1.5:
            result = True
        else:
            result = False
            liudongbi_date=0
    else:
        print("没有找到任何非空的流动比率")
        result = False
        liudongbi_date=0

    return result,liudongbi_date

def liudongbi_func(stock_code,stock_balance_sheet):
    first_row = stock_balance_sheet.iloc[0]
    # 提取需要的字段值
    report_date_name = first_row['REPORT_DATE_NAME']
    total_current_assets = first_row['TOTAL_CURRENT_ASSETS']
    total_liabilities = first_row['TOTAL_CURRENT_LIAB']
    liudongbi = total_current_assets / total_liabilities
    if liudongbi >= 1.5:
        result = True
    else:
        result = False
        liudongbi_date = 0
    # print(liudongbi,report_date_name)
    return result,report_date_name

def zhaiwu_lidong_func(stock_code,stock_balance_sheet,thresh=1.1):
    # 取第一行
    first_row = stock_balance_sheet.iloc[0]

    # 提取需要的字段值
    report_date_name = first_row['REPORT_DATE_NAME']
    total_current_assets = first_row['TOTAL_CURRENT_ASSETS']
    total_liabilities = first_row['TOTAL_LIABILITIES']

    # 计算指标
    debt_to_net_current_asset = total_liabilities / (total_current_assets - total_liabilities)
    # 打印结果
    # print(f"报告期：{report_date_name}")
    # print(f"流动资产总额（TOTAL_CURRENT_ASSETS）：{total_current_assets}")
    # print(f"总负债（TOTAL_LIABILITIES）：{total_liabilities}")
    print(f"总负债 / (流动资产 - 总负债)：{debt_to_net_current_asset}")
    if debt_to_net_current_asset <= thresh and debt_to_net_current_asset>0:
        result=True
    else:
        result=False
    return result

def youxingzichan(stock_code,full_code,spot_data,stock_balance_sheet,thresh):
    #现在市值
    target_row = spot_data[spot_data["代码"] == stock_code]
    total_market_value = target_row["总市值"].iloc[0]

    first_row = stock_balance_sheet.iloc[0]
    tangible_assets = first_row[[
        "FIXED_ASSET", "CIP", "OIL_GAS_ASSET",
        "PRODUCTIVE_BIOLOGY_ASSET", "INVENTORY",
        "MONETARYFUNDS", "USERIGHT_ASSET", "PROJECT_MATERIAL"
    ]].sum()
    print(f"total_market_value/tangible_assets：",total_market_value/tangible_assets)
    if total_market_value/tangible_assets < thresh:
        result = True
    else:
        result = False
    return result


def recent_dividend(stock_code,full_code,stock_balance_sheet):
    # 只保留最近5年的记录
    stock_fhps_detail_em_df = ak.stock_fhps_detail_em(symbol=stock_code)

    target_dates = ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31', '2020-12-31']
    # dates = ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31', '2020-12-31']
    #
    # # 转换为 datetime.date 格式
    # date_objects = pd.to_datetime(dates).date

    # 或者转换为 datetime 对象后，再取 .date
    date_objects = [d.date() for d in pd.to_datetime(target_dates)]
    df_selected = stock_fhps_detail_em_df[stock_fhps_detail_em_df["报告期"].isin(date_objects)]

    # 如果字段是“DIVIDEND_YIELD”，请替换成真实的列名
    print(df_selected[["报告期", "现金分红-股息率"]].sort_values("报告期", ascending=False).to_string(
        index=False))


stock_code = '300251'
# stock_name = row['名称']
spot_data = pd.read_csv('akshare_output/stocks.csv', encoding='utf-8-sig')
spot_data = format_stock_code(spot_data)
full_code = add_market_prefix(stock_code)

#2、流动比率>=1.5
result, liudongbi_date = liudongbi_func1(stock_code)
# result,liudongbi_date = liudongbi_func(stock_code,stock_balance_sheet)


#3、(债务/净流动资产价值)<=110%
stock_balance_sheet = ak.stock_balance_sheet_by_yearly_em(symbol=full_code)
result = zhaiwu_lidong_func(stock_code,stock_balance_sheet,thresh=1.1)


#4、有形资产
result = youxingzichan(stock_code,full_code,spot_data,stock_balance_sheet,thresh=1.3)

#5、过去5年分红是否都大于0
recent_dividend(stock_code,full_code,stock_balance_sheet)