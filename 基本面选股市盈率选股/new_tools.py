import akshare as ak
import pandas as pd
import time
import codecs

# 获取所有A股股票的基本信息
stock_info = ak.stock_info_a_code_name()

# 获取所有A股股票的市盈率、市净率等指标
stock_lg_indicator = ak.stock_a_lg_indicator()
# 去掉股票代码中的交易所前缀，方便后续合并
stock_lg_indicator['code'] = stock_lg_indicator['stock_code'].str[2:]

# 合并基本信息和指标信息
all_info = pd.merge(stock_info, stock_lg_indicator, on='code')

# 定义需要的年份和日期
income_dates = ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31', '2023-09-30', '2024-09-30']
cash_dates = ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2022-06-30', '2023-06-30', '2024-06-30']

# 定义获取股票代码的函数，便于获取行情数据
def get_stock_symbol(code):
    if code.startswith('6') or code.startswith('9'):
        return 'sh' + code
    else:
        return 'sz' + code

# 定义计算MACD的函数
def calculate_macd(data, short=12, long=26, m=9):
    data['DIFF'] = data['收盘'].ewm(adjust=False, alpha=2/(short+1), ignore_na=True).mean() - \
                   data['收盘'].ewm(adjust=False, alpha=2/(long+1), ignore_na=True).mean()
    data['DEA'] = data['DIFF'].ewm(adjust=False, alpha=2/(m+1), ignore_na=True).mean()
    data['MACD'] = 2 * (data['DIFF'] - data['DEA'])
    return data

# 定义检查净利润的函数
def income_check(code, dates):
    df = ak.stock_financial_report_sina(stock=code, symbol="利润表")
    # 转换日期格式
    df['报告日期'] = pd.to_datetime(df['报告日期'], format='%Y年%m月%d日')
    # 选取指定日期的数据
    df_selected = df[df['报告日期'].isin(pd.to_datetime(dates))]
    # 去除重复行
    df_selected = df_selected.drop_duplicates(subset='报告日期')
    # 将数据转换为字典
    dict_selected = df_selected.set_index('报告日期')['净利润'].to_dict()
    date_list = pd.to_datetime(dates)
    try:
        if (dict_selected[date_list[3]] - dict_selected[date_list[2]] > 0) and \
           (dict_selected[date_list[2]] - dict_selected[date_list[1]] > 0) and \
           (dict_selected[date_list[4]] - dict_selected[date_list[3]] > 0) and \
           (dict_selected[date_list[6]] - dict_selected[date_list[5]] > 0):
            return True
        else:
            return False
    except KeyError:
        return False

# 定义检查现金流的函数
def cash_check(code, dates):
    df = ak.stock_financial_report_sina(stock=code, symbol="现金流量表")
    # 转换日期格式
    df['报告日期'] = pd.to_datetime(df['报告日期'], format='%Y年%m月%d日')
    # 选取指定日期的数据
    df_selected = df[df['报告日期'].isin(pd.to_datetime(dates))]
    # 去除重复行
    df_selected = df_selected.drop_duplicates(subset='报告日期')
    # 将数据转换为字典
    dict_selected = df_selected.set_index('报告日期')['现金及现金等价物净增加额'].to_dict()
    opt_cashflow = df_selected.set_index('报告日期')['经营活动产生的现金流量净额'].to_dict()
    date_list = pd.to_datetime(dates)
    try:
        if dict_selected[date_list[3]] is not None and dict_selected[date_list[2]] is not None and \
           dict_selected[date_list[1]] is not None and dict_selected[date_list[5]] is not None:
            if (dict_selected[date_list[4]] - dict_selected[date_list[3]] > 0) and \
               (dict_selected[date_list[6]] - dict_selected[date_list[5]] > 0) and \
               opt_cashflow[date_list[6]] > 0 and \
               opt_cashflow[date_list[4]] > 0:
                print(f"股票代码：{code}, 满足现金流增长条件")
                return True
            else:
                return False
        else:
            return False
    except KeyError:
        return False

with codecs.open('底背离.csv', 'w', encoding='GBK') as f:
    # 对每支股票进行判断
    for index, row in all_info.iterrows():
        try:
            # 判断动态市盈率是否低于20
            if row['动态市盈率'] >= 20:
                continue
            code = row['code']
            # 检查净利润是否连续增长
            if not income_check(code, income_dates):
                continue
            # 检查现金流是否满足条件
            if not cash_check(code, cash_dates):
                continue
            # 获取股票的历史行情数据
            symbol = get_stock_symbol(code)
            df = ak.stock_zh_a_hist(symbol=symbol, adjust='qfq')
            # 计算MACD指标
            df = calculate_macd(df)
            # 定义观察窗口
            width = 10
            # 回溯周期
            lookback = 90
            df = df.sort_values(by='日期')
            df = df.reset_index(drop=True)
            # 找出MACD底背离的情况
            for i in range(1, len(df)):
                lowest_close = df['收盘'][max(0, i - lookback):i].min()
                lowest_macd = df['MACD'][max(0, i - lookback):i].min()
                if df['收盘'][i] < lowest_close and df['MACD'][i] > lowest_macd:
                    # 检查后续一段时间内是否突破最低价
                    for s in range(i + 1, min(i + width, len(df))):
                        if df['收盘'][s] > lowest_close:
                            print('股票代码：{}，股票名称：{}，日期：{}，出现底背离'.format(row['code'], row['name'], df['日期'][i]))
                            break
            # 将符合条件的股票写入文件
            f.write(row['code'] + ',' + row['name'] + '\n')
        except Exception as e:
            print(e)
            print("等待30秒")
            time.sleep(30)  # 等待30秒
