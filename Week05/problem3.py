import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from RiskMgmt import VaR, ES


# using the two csv files to get the daily total value of each portfolio, preparing for calculating returns
def parse_data(p, df):
    tot_value = pd.DataFrame(index=df.index, columns=["Total Value"])
    m = df.shape[0]

    # for data on each date
    for i in range(m):
        current = pd.DataFrame({"Stock":df.columns, "price":df.iloc[i,:]}) 
        #get the prices and holdings of each stock, calculate daily total price of all the holdings in a portfolio
        temp = pd.merge(current, p, on="Stock", how="inner")    
        temp_sum = sum(temp["price"] * temp["Holding"])
        tot_value.iloc[i][0] = temp_sum
    return tot_value


if __name__ == '__main__':

    df = pd.read_csv("DailyPrices.csv", index_col="Date")
    port = pd.read_csv("portfolio.csv").groupby("Portfolio")

    # total value of 3 portfolios on each day
    tot_port_value = pd.DataFrame(index=df.index, columns=["Total Port Value"])
    tot_port_value["Total Port Value"] = 0
    price_latest_value = [] # latest price of each portfolio

    # Arithmetric Return and T fitted VaR and ES
    # for each portfolio, get the daily total value, calculate Arithmetric Return
    rt_port_list = [] # return list of 3 portfolios
    for p in port:
        tot_value = parse_data(p[1], df)
        tot_port_value["Total Port Value"] += tot_value["Total Value"] # get the total value of 3 portfolios on each day
        price_latest_value.append(tot_value.iloc[-1]["Total Value"])
        
        # return for each of the 3 protfolios, mean = 0
        rt_ARITH = VaR.return_calculate(tot_value, "ARITH_RT")
        rt_ARITH = rt_ARITH.loc[:, "Total Value"]
        rt_ARITH = np.array(rt_ARITH[1:]).astype(np.float64)
        rt_ARITH = rt_ARITH - np.mean(rt_ARITH)
        rt_port_list.append(rt_ARITH)

    # return for total 3 portfolios, mean = 0
    rt_tot_ARITH = VaR.return_calculate(tot_port_value, "ARITH_RT")
    rt_tot_ARITH = rt_tot_ARITH.loc[:, "Total Port Value"]
    rt_tot_ARITH = np.array(rt_tot_ARITH[1:]).astype(np.float64)
    rt_tot_ARITH = rt_tot_ARITH - np.mean(rt_tot_ARITH)

    alpha = 0.05

    VaR_T, ES_T = ES.get_VaR_ES_T(rt_port_list[0], alpha)
    print("Portfolio A:")  
    print(f"VaR using a T distribution: ${price_latest_value[0] * VaR_T}, ES: ${price_latest_value[0] * ES_T}")
    VaR_T, ES_T = ES.get_VaR_ES_T(rt_port_list[1], alpha)
    print("Portfolio B:")  
    print(f"VaR using a T distribution: ${price_latest_value[1] * VaR_T}, ES: ${price_latest_value[1] * ES_T}")
    VaR_T, ES_T = ES.get_VaR_ES_T(rt_port_list[2], alpha)
    print("Portfolio C:")  
    print(f"VaR using a T distribution: ${price_latest_value[2] * VaR_T}, ES: ${price_latest_value[2] * ES_T}")
    VaR_T, ES_T = ES.get_VaR_ES_T(rt_tot_ARITH, alpha)
    print("VaR of total holdings")  
    print(f"VaR using a T distribution: ${sum(price_latest_value) * VaR_T}, ES: ${sum(price_latest_value) * ES_T}")
    

