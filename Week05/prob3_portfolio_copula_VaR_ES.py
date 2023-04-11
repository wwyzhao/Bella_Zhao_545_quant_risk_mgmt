import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
import scipy.stats as st
from RiskMgmt import VaR, ES, Simulation
import warnings
warnings.filterwarnings("ignore") 

def copula_t(rt, nsim = 1000):
    n = rt.shape[1]
    rt = (rt - rt.mean()).astype(np.float64)

    # calculate each T parameters, CDF for each column
    paras = [st.t.fit(rt[col]) for col in rt.columns]
    cdf = st.t.cdf(rt, *zip(*paras))
    cdf = pd.DataFrame(data=cdf, index=rt.index, columns=rt.columns)
    
    # convert from multivariate normal to simulations of a T distribution
    corr = cdf.corr(method = "spearman")
    sim_t = pd.DataFrame(st.norm.cdf(Simulation.simulate_pca(corr, nsim, 1)))
    sim_df = [st.t.ppf(sim_t[col], *zip(*paras)) for col in sim_t.columns]
    sim_df = pd.DataFrame(data=sim_df, columns=rt.columns)

    return sim_df
    

if __name__ == '__main__':

    df = pd.read_csv("DailyPrices.csv", index_col="Date")
    # current price of each stock
    cur_price = df.iloc[df.shape[0]-1,:]
    cur_price.index.name = "Stock"
    cur_price = pd.DataFrame({"price": cur_price}, index=cur_price.index)
    
    rt = VaR.return_calculate(df, "ARITH_RT")
    rt = rt[1:]
    nsim = 1000
    sim_df = copula_t(rt, nsim)
    
    # get simulated total value on the following 1000 days of each portfolio and all 3 portfolios
    ports = pd.read_csv("portfolio.csv").groupby("Portfolio")
    sum_port = pd.DataFrame()
    for name, port in ports:
        df_port = pd.merge(port, cur_price, on="Stock", how="inner")
        cur_value = sum(df_port["Holding"] * df_port["price"]) # current value
        
        sim_value = []
        for i in range(nsim):
            r = sim_df.iloc[i,:]
            r.index.name = "Stock"
            r = pd.DataFrame({"return": r}, index=r.index)
            df_temp = pd.merge(df_port, r, on="Stock", how="inner")
            fu_value = sum(df_temp["Holding"] * df_temp["price"] * (1 + df_temp["return"])) # future value using simulated return
            dif = fu_value - cur_value
            sim_value.append(dif)
        
        sum_port[name] = sim_value
    sum_port["total"] = sum_port["A"] + sum_port["B"] + sum_port["C"]
    
    for name, port in sum_port.iteritems():
        VaR_his, ES_his = ES.get_VaR_ES(port, alpha=0.05)
        print("Portfolio", name, "VaR: ", VaR_his, "ES: ", ES_his)
            
