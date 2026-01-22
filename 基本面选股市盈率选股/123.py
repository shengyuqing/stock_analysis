import requests
import pandas as pd


def washData(df):
    df = df.replace({0: '--'})  # 如果数据是0换成'--'

    # 数据第一列的中文会多出空格，需要处理掉
    df.columns.values[0] = df.columns.values[0].replace(' ', '')
    for row in range(0, df.shape[0]):
        col_name = df.columns.values[0]
        df.loc[row, col_name] = df.loc[row, col_name].replace(' ', '')


# number: 股票代码
# type: zycwzb:主要财务指标 zcfzb:资产负债表 lrb:利润表 xjllb:现金流表
def getReportData(number, type):
    url = 'http://quotes.money.163.com/service/' + type + '_' + number + '.html'
    f = open("temp.csv", "wb")
    f.write(requests.get(url).content)
    f.close()
    df = pd.read_csv('temp.csv', encoding='gbk')
    washData(df)

    dataList = []
    for col in range(1, df.shape[1] - 1):
        dataList.append({})
        col_name = df.columns.values[col]
        dataList[col - 1]['报告日期'] = col_name
        for row in range(0, df.shape[0]):
            key_name = df.loc[row, '报告日期']
            dataList[col - 1][key_name] = df.loc[row, col_name]

    return dataList


if __name__ == '__main__':
    print(getReportData('600000', 'zycwzb'))