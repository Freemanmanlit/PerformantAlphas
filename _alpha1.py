import numpy as np
import pandas as pd
from utils import EfficientAlpha as Alpha
from line_profiler import profile

class Alpha1(Alpha):

    def __init__(self,insts,dfs,start,end):
        super().__init__(insts,dfs,start,end)
    
    def pre_compute(self,trade_range):
        '''
        mean_12(
            neg(
                cszscre(
                    mult(
                        volume,
                        div(
                            minus(minus(close,low),minus(high,close)),
                            minus(high,low)
                        )
                    )
                )
            )
        )
        binary signal: 12-day mean cross-sectional z-score: x - mu(x) / std(x)
        '''
        
        self.op4s = {}
        for inst in self.insts:
            inst_df = self.dfs[inst]
            op1 = inst_df.volume
            op2 = (inst_df.close - inst_df.low) - (inst_df.high - inst_df.close)
            op3 = inst_df.high - inst_df.low
            op4 = op1 * op2 / op3       
            self.op4s[inst] = op4
        return
    
    @profile
    def post_compute(self,trade_range):
        temp = []
        for inst in self.insts:
            self.dfs[inst]["op4"] = self.op4s[inst]
            temp.append(self.dfs[inst]["op4"])

        # creating a unified data frame to compute cross sectional z-score
        temp_df = pd.concat(temp,axis=1)
        temp_df.columns = self.insts
        temp_df = temp_df.replace([np.inf, -np.inf], np.nan)
        zscore = lambda x: (x - np.nanmean(x))/np.nanstd(x)
        zscores = temp_df.ffill().apply(zscore, axis=1) # row-wise cs zscore
        cs_zscore_df =  zscores.rolling(12).mean() * -1
        
        alphas = []
        for inst in self.insts:
            inst_alpha = pd.DataFrame(cs_zscore_df[inst])
            self.dfs[inst]["alpha"] = inst_alpha.iloc[:, 0]
            alphas.append(self.dfs[inst]["alpha"])

        alphadf = pd.concat(alphas,axis=1)
        alphadf.columns = self.insts
        self.eligblesdf = self.eligiblesdf & (~pd.isna(alphadf)) # ensure alpha is non null
        self.alphadf = alphadf
        
        masked_df = self.alphadf/self.eligblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)
        num_eligibles = self.eligblesdf.sum(axis=1) # sum across columns
        rankdf= masked_df.rank(axis=1,method="average",na_option="keep",ascending=True)
        shortdf = rankdf.apply(lambda col: col <= num_eligibles.values/4, axis=0,raw=True)
        longdf = rankdf.apply(lambda col: col > np.ceil(num_eligibles - num_eligibles/4), axis=0, raw=True)
       
        forecast_df = -1*shortdf.astype(np.int32) + longdf.astype(np.int32)
        self.forecast_df = forecast_df
        
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values # .loc for row access
        return forecasts