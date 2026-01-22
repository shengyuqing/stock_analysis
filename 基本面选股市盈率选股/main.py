import time
import codecs
import pandas as pd


# 你需要定义calculate_rsi函数，假设你用的是一个简单的RSI计算方法
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 假设all_info是一个DataFrame，包含股票代码、公司信息等
# all_info = pd.read_csv('your_file.csv') # 示例加载数据

# 补充pro对象初始化，如果没有这个对象，需要初始化一个pro实例
# 假设你已经有了一个tushare的pro接口实例 'pro'
import tushare as ts
pro = ts.pro_api('dea5a5145efb8cd72300679ba0c3ea0942931ee3a5c6baf8d4dc443f')  # 请替换为你的Tushare API token

# 假设 income_dates 和 cash_dates 已经是你所需要的日期
income_dates = ['20211231', '20221231', '20231231', '20241231']
cash_dates = ['20211231', '20221231', '20231231', '20241231']

all_info = pro.stock_basic(
    fields='ts_code,name,industry,pe_ttm'
)

# 定义检查函数
def income_check(pro, dates):
    df = pro.income(ts_code=row['ts_code'], start_date='20191101', end_date='20251217',
                    fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,n_income')

    # 使用.loc函数选取满足条件的行
    df_selected = df[df['end_date'].isin(dates)]
    # 去除重复的行
    df_selected = df_selected.drop_duplicates()

    # 最后，我们可以将DataFrame转换为字典
    dict_selected = df_selected.set_index('end_date')['n_income'].to_dict()
    if (dict_selected['20221231'] - dict_selected['20211231'] > 0) and \
            (dict_selected['20211231'] - dict_selected['20201231'] > 0) and \
            (dict_selected['20231231'] - dict_selected['20221231'] > 0) and \
            (dict_selected['20241231'] - dict_selected['20231231'] > 0):
        return True
    else:
        return False

def cash_check(pro, cash_dates):
    df = pro.cashflow(ts_code=row['ts_code'], start_date='20181101', end_date='20231209',
                    fields='ts_code,ann_date,f_ann_date,end_date,im_net_cashflow_oper_act,end_bal_cash,beg_bal_cash,end_bal_cash_equ,beg_bal_cash_equ')

    # 使用.loc函数选取满足条件的行
    df_selected = df[df['end_date'].isin(cash_dates)]
    # 去除重复的行
    df_selected = df_selected.drop_duplicates()

    # 最后，我们可以将DataFrame转换为字典
    dict_selected = df_selected.set_index('end_date')['end_bal_cash'].to_dict()
    opt_cashflow = df_selected.set_index('end_date')['im_net_cashflow_oper_act'].to_dict()
    if dict_selected['20221231'] != None and dict_selected['20211231'] != None and dict_selected['20201231'] != None and dict_selected['20231231'] != None and dict_selected['20241231'] != None:
        if (dict_selected['20231231'] - dict_selected['20221231'] > 0) and \
           (dict_selected['20241231'] - dict_selected['20231231'] > 0) and \
           opt_cashflow['20241231'] > 0 and \
           opt_cashflow['20231231'] > 0:
            return True
        else:
            return False
    else:
        return False

def check_rsi(pro, row, threshold=35):
    df = ts.pro_bar(ts_code=row['ts_code'], adj='qfq', start_date='20220101', end_date='20231231')
    df['RSI'] = calculate_rsi(df)  # 计算RSI
    if df['RSI'].iloc[-1] < threshold:
        return True
    return False

# 写入文件
with codecs.open('底背离.csv', 'w', encoding='GBK') as f:
    for index, row in all_info.iterrows():
        try:
            if row['pe_ttm'] >= 20:  # 假设有市盈率条件
                continue
            if income_check(pro, income_dates) == False:
                continue
            if cash_check(pro, cash_dates) == False:
                continue
            if check_rsi(pro, row, threshold=35) == False:  # 判断RSI
                continue

            f.write(row['ts_code'] + ',' + row['name'] + ',' + row['industry'] + '\n')

        except Exception as e:
            print(e)
            print("等待70秒")
            time.sleep(70)  # 等待70秒
