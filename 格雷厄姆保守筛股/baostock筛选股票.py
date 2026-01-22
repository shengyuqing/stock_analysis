import baostock as bs
import pandas as pd
from datetime import datetime
import os

# 创建输出目录
output_dir = "output_csv"
os.makedirs(output_dir, exist_ok=True)

# ———— 1. 登录系统 ————
lg = bs.login()
if lg.error_code != '0':
    raise RuntimeError("BaoStock 登录失败：" + lg.error_msg)

# 统一取今日作为“最新交易日”
today = datetime.now().strftime("%Y-%m-%d")

# # ———— 2. 获取全部在市 A 股列表 ————
rs_stock = bs.query_all_stock(day=today)
# stocks = rs_stock.get_data()[['code', 'code_name']]
# stocks.to_csv(os.path.join(output_dir, 'step2_all_stocks.csv'), index=False)
stocks = pd.read_csv(os.path.join(output_dir, 'step2_all_stocks.csv'))

# # ———— 3. 获取“最新日频估值”指标：PE_TTM 和 收盘价 ————
# fields = "date,code,close,peTTM"
# pe_data = []
# for code in stocks['code']:
#     rs = bs.query_history_k_data_plus(
#         code, fields,
#         start_date=today, end_date=today,
#         frequency="d", adjustflag="1"
#     )
#     df = rs.get_data()
#     if not df.empty:
#         pe_data.append(df.iloc[-1])
# pe_df = pd.DataFrame(pe_data)
# pe_df[['close','peTTM']] = pe_df[['close','peTTM']].astype(float)
# pe_df.to_csv(os.path.join(output_dir, 'step3_pe.csv'), index=False)
# ———— 2. 读取已保存的 step3_pe.csv ————
pe_df = pd.read_csv(os.path.join(output_dir, 'step3_pe.csv'))


# # ———— 4. 获取“营运能力”中的流动比率（currentRatio） ————
# yr = datetime.now().year -1
# op_data = []
# for code in pe_df['code']:
#     rs = bs.query_operation_data(code=code, year=yr, quarter=4)
#     rows = []
#     # 必须用 rs.next() 一行行读，才能拿到所有字段
#     while rs.error_code == '0' and rs.next():
#         rows.append(rs.get_row_data())
#     # 这时 rows 是一个「N 行 × len(rs.fields) 列」的 list
#     df = pd.DataFrame(rows, columns=rs.fields)
#     if not df.empty:
#         df = df.astype({ 'currentRatio': float })
#         op_data.append(df.iloc[-1])
# op_df = pd.DataFrame(op_data)
# op_df.to_csv(os.path.join(output_dir, 'step4_current_ratio.csv'), index=False)

# ———— 4+5. 从资产负债表计算“流动比率” & 同时拿偿债数据 ————
yr = datetime.now().year -1
bs_data = []
for code in pe_df['code']:
    rs = bs.query_balance_data(code=code, year=yr, quarter=4)
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    tmp = pd.DataFrame(rows, columns=rs.fields)
    if tmp.empty:
        continue
    if tmp['currentRatio'].iloc[-1] == "":
        continue # 如果整行没有 currentRatio，就跳过这只股票

    # # 转数值
    # for f in ['totalLiability', 'totalCurrentAssets', 'totalCurrentLiability', 'totalShare', 'totalOwnersEquity']:
    #     if f in tmp:
    #         tmp[f] = tmp[f].astype(float)
    # # 计算流动比率
    # tmp['currentRatio'] = tmp['totalCurrentAssets'] / tmp['totalCurrentLiability']
    tmp['currentRatio'] = tmp['currentRatio'].astype(float)

    rs = bs.query_balance_sheet(code=code, year=yr, quarter=4)
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    raw_bs = pd.DataFrame(rows, columns=rs.fields)
    # 计算债务／净流动资产
    tmp['debt_to_netcur'] = tmp['totalLiability'] / (tmp['totalCurrentAssets'] - tmp['totalCurrentLiability']) * 100
    bs_data.append(tmp.iloc[-1])
bs_df = pd.DataFrame(bs_data)
bs_df.to_csv(os.path.join(output_dir, 'step4_balance_and_ratio.csv'), index=False)
#     df = pd.DataFrame(list(rs.get_data()), columns=rs.fields)
#     if not df.empty:
#         for f in ['totalLiability','totalCurrentAssets','totalCurrentLiability']:
#             df[f] = df[f].astype(float)
#         bs_data.append(df.iloc[-1])
# bs_df = pd.DataFrame(bs_data)
# bs_df.to_csv(os.path.join(output_dir, 'step5_balance.csv'), index=False)

# ———— 6. 合并并筛选条件 1–3 ————
df = pe_df.merge(bs_df[['code','currentRatio','debt_to_netcur','totalOwnersEquity','totalShare']], on='code')
cond1_3 = (
    (df['peTTM'] <= 9) &
    (df['currentRatio'] >= 1.5) &
    (df['debt_to_netcur'] <= 110)
)
df = df[cond1_3]
df.to_csv(os.path.join(output_dir, 'step6_cond1_3.csv'), index=False)

# ———— 7. 过去 5 年净利润 > 0 ————
profit_codes = []
years = list(range(yr-5, yr))
for code in df['code']:
    ok = True
    for y in years:
        rs = bs.query_profit_data(code=code, year=y, quarter=4)
        data = pd.DataFrame(list(rs.get_data()), columns=rs.fields)
        if data.empty or float(data['netProfit'][0]) <= 0:
            ok = False
            break
    if ok:
        profit_codes.append(code)
df = df[df['code'].isin(profit_codes)]
df.to_csv(os.path.join(output_dir, 'step7_profit.csv'), index=False)

# ———— 8. 过去 5 年股息 > 0 ————
div_codes = []
for code in df['code']:
    ok = True
    for y in years:
        rs = bs.query_dividend_data(code=code, year=str(y), yearType="report")
        data = pd.DataFrame(list(rs.get_data()), columns=rs.fields)
        if data.empty or data['dividCashPsBeforeTax'].astype(float).sum() <= 0:
            ok = False
            break
    if ok:
        div_codes.append(code)
df = df[df['code'].isin(div_codes)]
df.to_csv(os.path.join(output_dir, 'step8_dividend.csv'), index=False)

# ———— 9. 价格／有形资产 ≤ 120 ————
if 'intangibleAssets' in bs_df.columns and 'totalShare' in bs_df.columns:
    df = df.merge(bs_df[['code','intangibleAssets','totalShare']], on='code')
    df['tangible_per_share'] = (df['totalAssets'] - df['intangibleAssets']) / df['totalShare']
else:
    df = df.merge(bs_df[['code','totalOwnersEquity','totalShare']], on='code')
    df['tangible_per_share'] = df['totalOwnersEquity'] / df['totalShare']

df['price_to_tangible'] = df['close'] / df['tangible_per_share']
df = df[df['price_to_tangible'] <= 120]
df.to_csv(os.path.join(output_dir, 'step9_price_tangible.csv'), index=False)

# ———— 10. 最终结果 ————
result = df[['code','peTTM','currentRatio','debt_to_netcur','price_to_tangible']]
result.to_csv(os.path.join(output_dir, 'step10_result.csv'), index=False)
print("各步骤中间结果已保存至 output_csv 目录。")

# 登出
bs.logout()
