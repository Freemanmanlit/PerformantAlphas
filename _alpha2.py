import numpy as np
import pandas as pd
from utils import EfficientAlpha as Alpha

class Alpha2(Alpha):

    def __init__(self,insts,dfs,start,end):
        super().__init__(insts,dfs,start,end)
    
    def pre_compute(self,trade_range):
        '''
        continuous signal: mean_12(neg(minus(const_1,div(open,close))))
        '''                
        self.alphas = {}
        for inst in self.insts:
            inst_df = self.dfs[inst]
            alpha = -1 * (1-(inst_df.open/inst_df.close)).rolling(12).mean()
            self.alphas[inst] = alpha
        return
    
    def post_compute(self,trade_range):
        temp = []
        for inst in self.insts:
            self.dfs[inst]["alpha"] = self.alphas[inst]
            temp.append(self.dfs[inst]["alpha"])
        
        alphadf = pd.concat(temp,axis=1)
        alphadf.columns = self.insts
        alphadf = alphadf.ffill()
        self.eligblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        self.alphadf = alphadf
        return

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.alphadf.loc[date].values
        return forecasts
    