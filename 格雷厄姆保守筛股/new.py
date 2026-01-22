"""
author: ChatGPT (o3)
date  : 2025‑05‑19
desc  : 5 种选股策略示例，运行环境：
        pip install akshare pandas numpy
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime

# --- 基础数据拉取 -----------------------------------------------------------
def get_a_share_basics() -> pd.DataFrame:
    stock_basic = ak.stock_info_a_code_name()
    return stock_basic.rename(columns={"code": "ts_code"})

def get_finance(ts_code: str) -> pd.Series:
    df = ak.stock_financial_report_sina(stock=ts_code)
    df = df.set_index("report_date")
    latest = df.iloc[0]  # 最近一期
    return latest

# --- 通用指标拼装 -----------------------------------------------------------
def build_universe():
    basics = get_a_share_basics()
    records = []
    for _, row in basics.iterrows():
        code = row["ts_code"]
        try:
            fin = get_finance(code)
            records.append(
                {
                    "code"       : code,
                    "name"       : row["name"],
                    "pe"         : fin["pe_ttm"],
                    "pb"         : fin["pb_mrq"],
                    "div_yield"  : fin["dividendyield"] * 100,
                    "roe"        : fin["roe_diluted"],
                    "gross_margin": fin["grossprofitmargin"],
                    "debt_ratio" : fin["assetsdebtratio"],
                    "revenue_yoy": fin["operatingrevenue_yoy"],
                    "profit_yoy" : fin["netprofit_yoy"],
                    "rd_ratio"   : fin["rd-expensetor"](np.nan),
                    "ebit"       : fin["ebit"],
                    "ev"         : fin["enterprisevalue"],
                    "roic"       : fin["returnoninvestcapital"],
                }
            )
        except Exception:
            # 跳过无数据或上市未满一季的股票
            continue
    return pd.DataFrame(records)

# --- 策略函数 ---------------------------------------------------------------
def value_yield(df: pd.DataFrame):
    cond = (df["pe"] < 15) & (df["pb"] < 1.5) & (df["div_yield"] > 3)
    return df.loc[cond, ["code", "name", "pe", "pb", "div_yield"]]

def quality(df: pd.DataFrame):
    cond = (df["roe"] > 15) & (df["debt_ratio"] < 50) & (df["gross_margin"] > df["gross_margin"].quantile(0.7))
    return df.loc[cond, ["code", "name", "roe", "debt_ratio", "gross_margin"]]

def growth(df: pd.DataFrame):
    cond = (df["revenue_yoy"] > 20) & (df["profit_yoy"] > 25) & (df["rd_ratio"] > 5)
    return df.loc[cond, ["code", "name", "revenue_yoy", "profit_yoy", "rd_ratio"]]

def magic_formula(df: pd.DataFrame, top_n=30):
    df = df.dropna(subset=["ebit", "ev", "roic"])
    df["earn_yield_rank"] = df["ebit"] / df["ev"]
    df["earn_yield_rank"] = df["earn_yield_rank"].rank(ascending=False)
    df["roic_rank"] = df["roic"].rank(ascending=False)
    df["score"] = df["earn_yield_rank"] + df["roic_rank"]
    return df.nsmallest(top_n, "score")[["code", "name", "ebit", "ev", "roic", "score"]]

# --- 主流程 -----------------------------------------------------------------
if __name__ == "__main__":
    universe = build_universe()
    print("低估值+高分红：")
    print(value_yield(universe).head(20))

    print("\n高质量：")
    print(quality(universe).head(20))

    print("\n成长：")
    print(growth(universe).head(20))

    print("\nMagic Formula：")
    print(magic_formula(universe, top_n=20))
