fome datetime import date, timedelta
import numpy as np
import pandas as pd


class Deviation_Macd(object):
    def __init__(self,symbol):
        self.kline = self