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

    df = pd.read_csv("problem5.csv", index_col="Date")
    # current price of each stock
    cur_price = df.iloc[df.shape[0]-1,:]
    cur_price.index.name = "Stock"
    cur_price = pd.DataFrame({"price": cur_price}, index=cur_price.index)
    print(cur_price)

    rt = VaR.return_calculate(df, "ARITH_RT")
    rt = rt[1:]
    nsim = 1000
    # fit a generalized T distribution to each asset return series using a Gaussian Copula to get simulated returns
    sim_rtn = copula_t(rt, nsim)
    print(sim_rtn)
    
    # use simulated arithmetic returns to calculate simulated prices for each asset
    sim_prices = []
    for col in sim_rtn.columns:
        cur_p = cur_price.loc[col, "price"]
        sim_price = []
        for r in sim_rtn[col]:
            cur_p = cur_p * (1 + r)
            sim_price.append(cur_p)
        sim_prices.append(sim_price)
    # Calculate VaR (5%) for each asset
    VaR_assets = []
    for i in range(4):
        sim_r = np.array(sim_prices[i][1:]) - np.array(sim_prices[i][:-1])
        VaR_a, ES_a = ES.get_VaR_ES(sim_r, 0.05)
        VaR_assets.append(VaR_a)
    print("VaR (5%) for each asset")
    print(VaR_assets)

    # Calculate VaR (5%) for a portfolio of Asset 1 &2 and a portfolio of Asset 3&4
    sim_prices1 = []
    sim_prices2 = []
    for i in range(len(sim_prices[0])):
        sim_prices1.append(sim_prices[0][i] + sim_prices[1][i])
        sim_prices2.append(sim_prices[2][i] + sim_prices[3][i])
    # portfolio of Asset 1 &2
    sim_r_1 = np.array(sim_prices1[1:]) - np.array(sim_prices1[:-1])
    VaR_1_2, ES_1_2 = ES.get_VaR_ES(sim_r_1, 0.05)
    sim_r_2 = np.array(sim_prices2[1:]) - np.array(sim_prices2[:-1])
    VaR_3_4, ES_3_4 = ES.get_VaR_ES(sim_r_2, 0.05)
    print("VaR (5%) for a portfolio of Asset 1 &2")
    print(VaR_1_2)
    print("VaR (5%) for a portfolio of Asset 3 &4")
    print(VaR_3_4)
    
    # Calculate VaR (5%) for a portfolio of all 4 assets
    sim_prices_all = []
    for i in range(len(sim_prices[0])):
        sim_prices_all.append(sim_prices[0][i] + sim_prices[1][i] + sim_prices[2][i] + sim_prices[3][i])
    sim_r_all = np.array(sim_prices_all[1:]) - np.array(sim_prices_all[:-1])
    VaR_all, ES_all = ES.get_VaR_ES(sim_r_all, 0.05)
    print("VaR (5%) for a portfolio of all 4 assets")
    print(VaR_all)

