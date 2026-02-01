import numpy as np
import pandas as pd
from utils import EfficientAlpha as Alpha

class Alpha3(Alpha):

    def __init__(self,insts,dfs,start,end):
        super().__init__(insts,dfs,start,end)
    
    def pre_compute(self,trade_range):
        for inst in self.insts:
            inst_df = self.dfs[inst]
            fast = np.where(inst_df.close.rolling(10).mean() > inst_df.close.rolling(50).mean(), 1, 0)
            medium = np.where(inst_df.close.rolling(20).mean() > inst_df.close.rolling(100).mean(), 1, 0)
            slow = np.where(inst_df.close.rolling(50).mean() > inst_df.close.rolling(200).mean(), 1, 0)
            alpha = fast + medium + slow
            self.dfs[inst]["alpha"] = alpha
        return
    
    def post_compute(self,trade_range):
        temp = []
        for inst in self.insts:
            temp.append(self.dfs[inst]["alpha"])
        alphadf = pd.concat(temp,axis=1)
        alphadf.columns = self.insts
        alphadf = alphadf.ffill()
        self.eligblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        self.alphadf = alphadf
        return

    def compute_signal_distribution(self, eligibles, date):
        return self.alphadf.loc[date]
    