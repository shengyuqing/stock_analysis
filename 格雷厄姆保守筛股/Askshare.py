import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


def get_qualified_stocks():
    # 获取所有A股实时行情数据（包含动态市盈率）
    spot_data = ak.stock_zh_a_spot_em()
    spot_data.to_csv('askshare_output/stocks.csv', index=False)
    if spot_data.empty:
        print("无法获取A股实时行情数据")
        return pd.DataFrame()

        # 筛选动态市盈率<=9的股票
    filtered = spot_data[(spot_data['市盈率-动态'] <= 9) & (spot_data['市盈率-动态'] > 0)]

    qualified_stocks = []

    for _, row in filtered.iterrows():
        stock_code = row['代码']
        stock_name = row['名称']

        try:
            # 获取财务指标数据
            financial = ak.stock_financial_analysis_indicator(symbol=stock_code)
            if financial.empty:
                continue

                # 条件2：流动比率>=1.5
            current_ratio = financial['流动比率'].iloc[0]
            if current_ratio < 1.5:
                continue

                # 条件3：(债务/净流动资产价值)<=110%
            total_debt = financial['负债合计'].iloc[0]
            current_assets = financial['流动资产合计'].iloc[0]
            current_liabilities = financial['流动负债合计'].iloc[0]
            net_current_asset_value = current_assets - current_liabilities
            if net_current_asset_value <= 0:
                continue
            debt_to_ncav = (total_debt / net_current_asset_value) * 100
            if debt_to_ncav > 110:
                continue

                # 条件4：过去5年净利润>0
            current_year = datetime.now().year
            all_positive_net_profit = True
            for year in range(current_year - 1, current_year - 6, -1):
                try:
                    report = ak.stock_financial_report_sina(stock_code, f"{year}年报")
                    if report.empty or report['净利润(万元)'].iloc[0] <= 0:
                        all_positive_net_profit = False
                        break
                except:
                    all_positive_net_profit = False
                    break
            if not all_positive_net_profit:
                continue

                # 条件5：过去5年股息>0
            all_positive_dividend = True
            for year in range(current_year - 1, current_year - 6, -1):
                try:
                    dividend = ak.stock_history_dividend_detail(stock_code)
                    if dividend.empty or dividend[dividend['公告日期'].str.contains(str(year))][' 现金分红'].iloc[
                        0] <= 0:
                        all_positive_dividend = False
                        break
                except:
                    all_positive_dividend = False
                    break
            if not all_positive_dividend:
                continue

                # 条件6：价格/有形资产<=120%
            tangible_assets = financial['有形资产(万元)'].iloc[0]
            if tangible_assets <= 0:
                continue
            price = row['最新价']
            price_to_tangible = (price / (tangible_assets / 10000)) * 100  # 单位：万元->元
            if price_to_tangible > 120:
                continue

                # 添加到结果列表
            qualified_stocks.append({
                '代码': stock_code,
                '名称': stock_name,
                '市盈率-动态': row['市盈率-动态'],
                '流动比率': current_ratio,
                '债务/净流动资产价值(%)': round(debt_to_ncav, 2),
                '价格/有形资产(%)': round(price_to_tangible, 2),
                '最新价': row['最新价']
            })
            print(f"找到符合条件的股票: {stock_code} {stock_name}")

        except Exception as e:
            print(f"处理股票 {stock_code} 时出错: {str(e)}")
            continue

    return pd.DataFrame(qualified_stocks)


# 执行筛选
result = get_qualified_stocks()

# 保存结果
if not result.empty:
    result.to_excel(" 符合条件的A股股票.xlsx", index=False)
    print(f"共找到 {len(result)} 只符合条件的股票")
    print(result)
else:
    print("没有找到符合条件的股票")